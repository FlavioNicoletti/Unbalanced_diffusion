#!/usr/bin/env python3
"""
================================================================================
 Probing the "inference" of the class centroids by a random-feature model
 of the diffusion score (Gaussian mixture)
================================================================================

OUTLINE OF THE SCRIPT
---------------------
This script studies the *inference transition* of a random-feature model built
on a two-cluster Gaussian mixture — a minimal model of *class imbalance* in
diffusion generative models. As we scan the diffusion time backwards, the
empirical feature covariance `U` develops isolated top eigenvalues whose
eigenvectors start to *align with a single cluster centroid*: the random-feature
model "infers" the class centroids. That alignment ("inference") is the object
we probe here. It is a property of the random-feature model, distinct from the
speciation of the diffusion process (whose time scale sets the probed window,
see section 1).

For several random realizations and, within each, a sweep of diffusion times we:

  1. Draw a synthetic dataset of `n_samples` points in dimension `d`, split into
     two Gaussian clusters with tunable mixing weight, variances and centroid
     length scales. Strong under-sampling of the minority class can be requested
     via an exponent (weight = 1 - d^{-exponent}).
  2. Build a random projection matrix W (the "random features", shape p x d).
  3. Compute per-cluster "Hermite integrals" (alpha, beta_tilde, gamma_tilde,
     beta) that describe how the activation function (tanh by default; chosen
     with -act from the ACTIVATIONS dictionary) acts on Gaussian inputs, using
     fast 1D/2D trapezoidal numerical integration (JIT-compiled with numba). For
     non-zero centroids the coefficients differ across feature dimensions, so
     each cluster yields a (p x 4) array.
  4. Assemble the empirical feature covariance `U` (sample-based), summing the
     per-sample contributions in parallel BATCHES (joblib).
  5. Diagonalize `U` and, at each diffusion time, record:
        - the top three eigenvalues of `U`,
        - the cosine similarity of the top two eigenvectors with each projected
          cluster centroid,
        - the average cosine similarity of all remaining eigenvectors with each
          projected centroid.
  6. Append these "inference signals" to a text file under `results_inference/`.

The heavy Hermite integrals are JIT-compiled with numba (2D trapezoidal
integration for `beta`, ~3000x faster than Monte-Carlo at comparable accuracy),
and the per-sample loop is split into batches that run in parallel via joblib.

--------------------------------------------------------------------------------
HOW TO RUN THE SCRIPT
--------------------------------------------------------------------------------
Basic run with defaults:

    python centroids_SNR_at_speciation_scale.py

Custom run (d=100, p=800 features, 400 samples, 10 realizations, batch size 50):

    python centroids_SNR_at_speciation_scale.py -d 100 -p 800 -s 400 -r 10 -b 50

Same run with ReLU activation:

    python centroids_SNR_at_speciation_scale.py -d 100 -p 800 -s 400 -r 10 -b 50 -act relu

Strong under-sampling of the minority class (weight = 1 - d^{-0.5}):

    python centroids_SNR_at_speciation_scale.py -u_samp_exp 0.5

See all options:

    python centroids_SNR_at_speciation_scale.py --help

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    numpy, numba, joblib

--------------------------------------------------------------------------------
STRUCTURE OF THE OUTPUT FOLDER
--------------------------------------------------------------------------------
A nested folder is created under `results_inference/` whose name encodes the diffusion
type, the centroid factors and the activation, e.g.:

    results_inference/output_spectral_quantities_sampling_bias_probe_inference_type=OU_centroid1=3.0_centroid2=3.0_activation=tanh/
        P=800_D=100_S=400/
            signals_to_probe_inference_weight=0.5_var1=0.5_var2=0.5_ortho=False_batch.txt

Each row of the file corresponds to one diffusion time and contains (in order):

    t   lambda_1  lambda_2  lambda_3
        cos(v1, mu_1)  cos(v1, mu_2)  cos(v2, mu_2)  cos(v2, mu_1)
        <cos(other eigvecs, mu_1)>   <cos(other eigvecs, mu_2)>

where lambda_1 >= lambda_2 >= lambda_3 are the top three eigenvalues of U,
v1 / v2 are the eigenvectors of the largest / second-largest eigenvalue, and
mu_1 / mu_2 are the projected cluster centroids.
================================================================================
"""

import os
import time
import secrets
import argparse

import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange

# Record wall-clock start so we can report the total runtime at the end.
start_time = time.time()


# =============================================================================
# 1. COMMAND-LINE ARGUMENTS
# -----------------------------------------------------------------------------
# Every knob of the experiment is exposed here so you can sweep parameters from
# the shell without editing the file. Run with --help to see them all.
# =============================================================================

# --- Available activation functions (selected with -act / --activation) -----
# To use another activation, add an entry here: its Hermite coefficients are
# evaluated numerically, so nothing else needs to change. The function must
# work on both scalars and numpy arrays and be compilable by numba (plain numpy
# expressions such as np.tanh, np.maximum, np.exp, ... are fine).
ACTIVATIONS = {
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(x, 0.0),
}

parser = argparse.ArgumentParser(
    description="Probe when a random-feature model of the diffusion score infers the class "
                "centroids (inference transition), for a Gaussian mixture "
                "(batch processing + fast trapezoidal integration).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# --- Problem size -----------------------------------------------------------
parser.add_argument("-d", "--dimension", type=int, default=100,
                    help="Ambient dimension d of each data point.")
parser.add_argument("-p", "--projections", type=int, default=800,
                    help="Number p of random features / projections.")
parser.add_argument("-s", "--samples", type=int, default=400,
                    help="Total number of samples drawn per realization.")
parser.add_argument("-r", "--realizations", type=int, default=10,
                    help="How many independent random realizations to run.")

# --- Centroid geometry ------------------------------------------------------
parser.add_argument("--ortho", action="store_true",
                    help="Use orthogonal centroids (cosine = 0). (Kept for interface "
                         "compatibility; it only affects the output file name here.)")

# --- Parallelism ------------------------------------------------------------
parser.add_argument("-b", "--batch_size", type=int, default=50,
                    help="Number of samples processed together in one parallel batch.")

# --- Gaussian mixture parameters -------------------------------------------
parser.add_argument("-w", "--weight", type=float, default=0.5,
                    help="Mixing weight of cluster 1 (cluster 2 gets 1 - weight). "
                         "Ignored when --under_sampling_exponent > 0.")
parser.add_argument("-var1", "--var1", type=float, default=0.5,
                    help="Variance of cluster 1.")
parser.add_argument("-var2", "--var2", type=float, default=0.5,
                    help="Variance of cluster 2.")
parser.add_argument("-c1", "--centroid_factor1", type=float, default=3.0,
                    help="Length scale of centroid 1.")
parser.add_argument("-c2", "--centroid_factor2", type=float, default=3.0,
                    help="Length scale of centroid 2.")

# --- Under-sampling of the minority class -----------------------------------
# When > 0, the majority-class weight is set to 1 - d^{-exponent}, so the
# minority class is strongly under-sampled as d grows (a specific imbalance
# scaling). When 0, the fixed --weight value is used instead.
parser.add_argument("-u_samp_exp", "--under_sampling_exponent", type=float, default=0.0,
                    help="Exponent for strong under-sampling of the minority class "
                         "(0.0 = use the fixed --weight; must be non-negative).")

# --- Activation function ----------------------------------------------------
parser.add_argument("-act", "--activation", type=str.lower, default="tanh",
                    choices=list(ACTIVATIONS),
                    help="Activation function of the random features (keys of ACTIVATIONS).")

args = parser.parse_args()

# --- Resolve parameters from the command line -------------------------------
d = args.dimension
p = args.projections
n_samples = args.samples
n_realizations = args.realizations
orthogonal_centroids = args.ortho
batch_size = args.batch_size
under_sampling_exponent = args.under_sampling_exponent
activation = args.activation

# JIT-compile the chosen activation so that the numba integrators below can call it.
phi = njit(ACTIVATIONS[activation])

if under_sampling_exponent > 0:
    # Majority-class weight chosen so the minority class is strongly under-sampled.
    weight = 1.0 - d ** (-under_sampling_exponent)
elif under_sampling_exponent == 0:
    weight = args.weight
else:
    raise ValueError("under_sampling_exponent must be non-negative")

var1 = args.var1
var2 = args.var2
centroids_factor1 = args.centroid_factor1
centroids_factor2 = args.centroid_factor2

# Mixing weights of the two clusters.
weight1 = weight
weight2 = 1.0 - weight1

# --- Diffusion-time sweep ---------------------------------------------------
# We scan the diffusion time from t_ini down to t_fin (n_times steps). The range
# is set relative to log(d), the d-dependent time scale of speciation of the
# diffusion process, around which we probe when the random-feature model infers
# the class centroids. Change these to move/zoom the probed window.
t_ini = 0.8 * np.log(d)
t_fin = 0.1 * np.log(d)
n_times = 20
t_diffusion = np.linspace(t_ini, t_fin, n_times)
diffusion_type = "OU"  # "FD" (Free Diffusion) or "OU" (Ornstein-Uhlenbeck)

n1 = int(n_samples * weight1)
n2 = n_samples - n1

# --- Random number generator ------------------------------------------------
# Seeded from OS entropy, so runs differ by default. The seed is printed below;
# replace it with a fixed integer for a fully reproducible run.
seed = secrets.randbits(128)
rng = np.random.default_rng(seed)

print(f"Parameters: d={d}, p={p}, n_samples={n_samples}, n_realizations={n_realizations}, ortho={orthogonal_centroids}, activation={activation}")
print(f"Batch size: {batch_size}, weight: {weight}, variance cluster 1: {var1}, variance cluster 2: {var2}")
print("seed = ", seed)

# Diffusion coefficients a_t (signal scaling) and Delta_t (noise variance) at
# each swept time. Vectorized over t_diffusion.
if diffusion_type == "FD":       # Free Diffusion: signal kept, noise grows linearly.
    adiff = 1.0
    Delta_t = t_diffusion
elif diffusion_type == "OU":     # Ornstein-Uhlenbeck: signal decays, variance -> 1.
    adiff = np.exp(-t_diffusion)
    Delta_t = 1.0 - np.exp(-2.0 * t_diffusion)


# =============================================================================
# 2. INTEGRATION GRID FOR THE TRAPEZOIDAL RULE
# -----------------------------------------------------------------------------
# The Hermite integrals below are Gaussian expectations of phi(...). We compute
# them by trapezoidal quadrature on a fixed grid xi in [-xi_max, xi_max], with
# the Gaussian weight exp(-xi^2/2) precomputed. Increase n_grid / xi_max for
# higher accuracy at the cost of speed.
# =============================================================================
n_grid = 2000            # number of grid points for integration
xi_max = 10.0            # integration range: [-xi_max, xi_max]
xi = np.linspace(-xi_max, xi_max, n_grid)
gaussian_weights = np.exp(-xi ** 2 / 2.0)   # exp(-x^2/2), the (unnormalized) Gaussian weight

print(f"Integration grid: {n_grid} points on [-{xi_max}, {xi_max}]")
print(f"Using 2D trapezoidal integration for beta (fast & accurate, ~3000x faster than MC)")


# =============================================================================
# 3. DATA GENERATION
# -----------------------------------------------------------------------------
# Draw a two-cluster isotropic Gaussian mixture. The two centroids are built to
# have a prescribed cosine similarity `r`: start from the anti-aligned pair
# (+c1..+c1) and (-c2..-c2) and flip the sign of a fraction of the coordinates
# of the second centroid so the overlap matches `r`. (Here `r` is called with 0.)
# =============================================================================
def generate_dataset(n, d, weight_cluster_1, vars, centroids_factor_1, centroids_factor_2, r):
    n_1 = int(weight_cluster_1 * n)   # samples in cluster 1
    n_2 = n - n_1                     # samples in cluster 2

    var_1 = vars[0]
    var_2 = vars[1]

    # Spatially homogeneous centroids with the requested cosine similarity.
    m_1 = centroids_factor_1 * np.ones(d)
    m_2 = -centroids_factor_2 * np.ones(d)
    if d % 2 == 0:
        ind = int(r * d // 2)
        m_2[d // 2 - ind:] *= -1.0
    else:
        ind = int(r * (d - 1) // 2)
        m_2[d // 2 - ind:d // 2 + ind + 1] *= -1.0

    # Isotropic Gaussian samples around each centroid, standard deviation sqrt(var).
    cluster_1_samples = rng.standard_normal((n_1, d)) * np.sqrt(var_1) + m_1
    cluster_2_samples = rng.standard_normal((n_2, d)) * np.sqrt(var_2) + m_2

    # Guard against an empty cluster under strong under-sampling.
    if n_1 == 0:
        dataset = cluster_2_samples
    elif n_2 == 0:
        dataset = cluster_1_samples
    else:
        dataset = np.vstack([cluster_1_samples, cluster_2_samples])

    return [dataset, m_1, m_2]


# =============================================================================
# 4. HERMITE INTEGRALS (numba-accelerated)
# -----------------------------------------------------------------------------
# These are the Gaussian expectations that characterize how the activation phi
# (a numba-compiled function, see ACTIVATIONS) acts on the projected inputs. They
# are the heaviest part of the computation and are JIT-compiled with numba. The
# two kernels taking phi as an argument are not cached on disk (numba cannot
# reliably cache functions that take other compiled functions as arguments), so
# they are compiled at every run (about one second).
# =============================================================================
@njit(cache=True)
def trapz_1d(y, x):
    """Numba-compatible implementation of numpy.trapz for 1D arrays."""
    n = y.shape[0]
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s


@njit(parallel=True)
def compute_beta_trapz_2d(phi, mu, var_sample, adiff, Delta_t, p, xi, gaussian_weights):
    """2D trapezoidal integration for the `beta` coefficient.

    beta[i] = ∫ p(xi1) * [ ∫ phi(scale1*xi1 + scale2*xi2 + shift_i) p(xi2) dxi2 ]^2 dxi1

    This is ~3000x faster than Monte-Carlo with 1e8 samples while achieving
    < 1e-4 accuracy. Parallelized over the p feature dimensions.
    """
    beta_vals = np.zeros(p)
    n_grid = len(xi)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    scale_1 = np.sqrt(var_sample) * adiff   # scaling of the "signal" Gaussian
    scale_2 = np.sqrt(Delta_t)              # scaling of the "noise" Gaussian

    # Normalized Gaussian PDF values on the grid.
    gaussian_pdf = gaussian_weights * inv_sqrt_2pi

    for i in prange(p):
        shift_i = adiff * mu[i]
        integrand_xi1 = np.zeros(n_grid)

        # Outer integral over xi1.
        for idx1 in range(n_grid):
            # Inner integral over xi2.
            integral_xi2_vals = np.zeros(n_grid)
            for idx2 in range(n_grid):
                noise = scale_1 * xi[idx1] + scale_2 * xi[idx2] + shift_i
                integral_xi2_vals[idx2] = phi(noise) * gaussian_pdf[idx2]

            # Integrate over xi2 (trapezoidal rule).
            sum_xi2 = 0.0
            for j in range(n_grid - 1):
                sum_xi2 += 0.5 * (integral_xi2_vals[j] + integral_xi2_vals[j + 1]) * (xi[j + 1] - xi[j])

            # Square the inner integral and weight by p(xi1).
            integrand_xi1[idx1] = (sum_xi2 ** 2) * gaussian_pdf[idx1]

        # Integrate over xi1 (trapezoidal rule).
        sum_xi1 = 0.0
        for i1 in range(n_grid - 1):
            sum_xi1 += 0.5 * (integrand_xi1[i1] + integrand_xi1[i1 + 1]) * (xi[i1 + 1] - xi[i1])

        beta_vals[i] = sum_xi1

    return beta_vals


@njit(parallel=True)
def compute_hermite_integrals_GEP_hybrid(phi, Gamma_t, mu, var_sample, adiff, Delta_t, p, xi, gaussian_weights):
    """Compute the four Hermite integrals per feature dimension.

    For the activation phi and each dimension i (with shift_i = adiff * mu[i] and z ~ N(0,1)):
        alpha[i]       = E[ phi(Gamma_t*z + shift_i) ]             (1D trapz)
        beta_tilde[i]  = E[ phi^2(Gamma_t*z + shift_i) ]           (1D trapz)
        gamma_tilde[i] = E[ phi(Gamma_t*z + shift_i) * z ]         (1D trapz)
        beta[i]        = correlated 2D expectation (see compute_beta_trapz_2d)

    Returns an array of shape (p, 4) with columns [alpha, beta_tilde, gamma_tilde, beta].
    """
    hermite_integrals = np.zeros((p, 4))
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    # ---- alpha, beta_tilde, gamma_tilde: three 1D integrals per dimension ----
    for i in prange(p):
        shift_i = adiff * mu[i]
        phi_vals = phi(Gamma_t * xi + shift_i)

        # alpha: E[phi(...)]
        integrand_alpha = phi_vals * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 0] = trapz_1d(integrand_alpha, xi)

        # beta_tilde: E[phi^2(...)]
        integrand_beta_tilde = phi_vals ** 2 * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 1] = trapz_1d(integrand_beta_tilde, xi)

        # gamma_tilde: E[phi(...) * z]
        integrand_gamma_tilde = phi_vals * xi * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 2] = trapz_1d(integrand_gamma_tilde, xi)

    print(f"  ✓ Computed alpha, beta_tilde, gamma_tilde (1D trapz).")

    # ---- beta: the correlated 2D integral ----
    hermite_integrals[:, 3] = compute_beta_trapz_2d(phi, mu, var_sample, adiff, Delta_t, p, xi, gaussian_weights)
    print(f"  ✓ Computed beta (2D trapz).")

    return hermite_integrals


def print_hermite_integrals(hermite_integrals, fname, p):
    """(Optional) Write the Hermite integrals of one cluster to a text file.

    Not called by default; the calls in the main loop are commented out. Enable
    them if you want to inspect the coefficients."""
    with open(fname, "a") as f:
        print("# Index\talpha\tbeta_tilde\tgamma_tilde\tbeta", file=f)
        for i in range(p):
            print(i, hermite_integrals[i, 0], hermite_integrals[i, 1],
                  hermite_integrals[i, 2], hermite_integrals[i, 3], file=f)


# =============================================================================
# 5. EMPIRICAL FEATURE COVARIANCE U (per-batch contribution)
# -----------------------------------------------------------------------------
# The per-sample contributions to the empirical feature covariance `U` are summed
# here. This function is called in parallel for different batches of samples;
# each batch processes its samples sequentially and returns the partial sum,
# which keeps memory low compared to materializing every per-sample matrix.
# The Hermite integrals are PRECOMPUTED per cluster (only 2 clusters), so each
# sample just selects its cluster's coefficients.
# =============================================================================
def process_batch(batch_indices, X, W, cluster_indices,
                  hermite_integrals_cluster1, hermite_integrals_cluster2,
                  mu_lambda_cluster1, mu_lambda_cluster2,
                  var1, var2, Gamma_t1, Gamma_t2,
                  adiff, Delta_t, p, n_samples, d_dim,
                  batch_id=None, checkpoint_interval=10):
    """Return the summed U contribution of the samples in `batch_indices`.

    Args:
        batch_indices: sample indices handled by this batch.
        X: all samples, shape (n_samples, d).
        W: random features, shape (p, d).
        cluster_indices: cluster label (0 or 1) per sample, shape (n_samples,).
        hermite_integrals_cluster1/2: precomputed (p, 4) coefficients per cluster.
        mu_lambda_cluster1/2: projected centroids W @ centroid / sqrt(d), shape (p,).
        var1, var2: cluster variances.
        Gamma_t1, Gamma_t2: sqrt(var_c * adiff^2 + Delta_t) per cluster.
        adiff, Delta_t: diffusion parameters at this time.
        p, n_samples, d_dim: sizes.
        batch_id, checkpoint_interval: progress-logging controls.

    Returns:
        U_GEP_batch: (p, p) sum of the batch's per-sample contributions.
    """
    U_GEP_batch = np.zeros((p, p))
    batch_size_actual = len(batch_indices)

    # Internal RNG per batch, seeded independently, to avoid contention across
    # parallel workers.
    seed_batch = secrets.randbits(128)
    rng = np.random.default_rng(seed_batch)

    # W @ W.T is shared across all samples in this batch.
    WWT = np.dot(W, W.T)

    # gamma (per cluster) is derived from the precomputed gamma_tilde.
    gamma_tilde_1 = hermite_integrals_cluster1[:, 2]
    gamma_tilde_2 = hermite_integrals_cluster2[:, 2]
    gamma_cluster1 = gamma_tilde_1 * np.sqrt(var1) * adiff / Gamma_t1
    gamma_cluster2 = gamma_tilde_2 * np.sqrt(var2) * adiff / Gamma_t2

    if batch_id is not None:
        print(f"    [Batch {batch_id}] Starting - {batch_size_actual} samples "
              f"(indices {batch_indices[0]}-{batch_indices[-1]})")

    samples_processed = 0
    for n in batch_indices:
        X_n = X[n, :]
        cluster_idx = cluster_indices[n]

        # Select the precomputed quantities for this sample's cluster.
        if cluster_idx == 0:
            hermite_integrals = hermite_integrals_cluster1
            mu_lambda = mu_lambda_cluster1
            var_sample = var1
            gamma = gamma_cluster1
        else:
            hermite_integrals = hermite_integrals_cluster2
            mu_lambda = mu_lambda_cluster2
            var_sample = var2
            gamma = gamma_cluster2

        # Projected sample Lambda = W @ X_n / sqrt(d).
        Lambda = np.dot(W, X_n) / np.sqrt(d_dim)

        alpha = hermite_integrals[:, 0]
        beta_tilde = hermite_integrals[:, 1]
        beta = hermite_integrals[:, 3]

        # Random noise realizing the fluctuating part of the feature vector.
        eta = rng.standard_normal(size=p)

        # Variance term, clamped to be non-negative (guards tiny negative values
        # from numerical error).
        variance_term = np.maximum(beta - gamma ** 2 - alpha ** 2, 0.0)

        # Feature vector for this sample and its rank-1 covariance contribution.
        GEP_vector = alpha + gamma * (Lambda - mu_lambda) / np.sqrt(var_sample) + np.sqrt(variance_term) * eta
        U_GEP_sample = np.outer(GEP_vector, GEP_vector) / n_samples

        # W @ W.T term.
        U_GEP_sample += Delta_t / (var_sample * adiff ** 2 * d_dim) * WWT * np.outer(gamma, gamma) / n_samples

        # Diagonal correction.
        diag_correction = (beta_tilde - beta - Delta_t / (var_sample * adiff ** 2) * gamma ** 2) / n_samples
        U_GEP_sample += np.diag(diag_correction)

        U_GEP_batch += U_GEP_sample

        samples_processed += 1
        if batch_id is not None and (samples_processed % checkpoint_interval == 0
                                     or samples_processed == batch_size_actual):
            print(f"    [Batch {batch_id}] Processed {samples_processed}/{batch_size_actual} samples "
                  f"({100 * samples_processed / batch_size_actual:.0f}%)")

    if batch_id is not None:
        print(f"    [Batch {batch_id}] Complete ✓")

    return U_GEP_batch


def evaluate_U_tilde(random_features, hermite_coefficients1, hermite_coefficients2):
    """(Optional) Population / generalization counterpart of U (sample-free).

    Not used to produce the inference signals below, but kept as a reference
    implementation: it assembles the mixture-averaged population feature
    covariance directly from the Hermite coefficients of the two clusters.
    """
    alpha1 = hermite_coefficients1[:, 0]
    beta_tilde1 = hermite_coefficients1[:, 1]
    gamma_tilde1 = hermite_coefficients1[:, 2]

    alpha2 = hermite_coefficients2[:, 0]
    beta_tilde2 = hermite_coefficients2[:, 1]
    gamma_tilde2 = hermite_coefficients2[:, 2]

    wishartlike_1 = random_features @ random_features.T / random_features.shape[1] * (np.outer(gamma_tilde1, gamma_tilde1))
    wishartlike_2 = random_features @ random_features.T / random_features.shape[1] * (np.outer(gamma_tilde2, gamma_tilde2))

    diagonal1 = beta_tilde1 - gamma_tilde1 ** 2 - alpha1 ** 2
    diagonal2 = beta_tilde2 - gamma_tilde2 ** 2 - alpha2 ** 2

    centroids_pert1 = np.outer(alpha1, alpha1)
    centroids_pert2 = np.outer(alpha2, alpha2)

    Utilde1 = wishartlike_1 + np.diag(diagonal1) + centroids_pert1
    Utilde2 = wishartlike_2 + np.diag(diagonal2) + centroids_pert2

    Utilde = weight1 * Utilde1 + weight2 * Utilde2

    return Utilde, Utilde1, Utilde2


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    # Output folder encodes the diffusion type, centroid factors, activation and sizes.
    # (Results are appended, so runs with different activations must not share a folder.)
    output_dir = (f"results_inference/output_spectral_quantities_sampling_bias_probe_inference_"
                  f"type={diffusion_type}_centroid1={centroids_factor1}_centroid2={centroids_factor2}_"
                  f"activation={activation}/"
                  f"P={p}_D={d}_S={n_samples}/")
    os.makedirs(output_dir, exist_ok=True)

    # Output file (opened in append mode). The name encodes whether fixed weight
    # or an under-sampling exponent was used.
    if under_sampling_exponent == 0:
        fname = open(os.path.join(
            output_dir,
            f"signals_to_probe_inference_weight={weight1}_var1={var1}_var2={var2}_ortho={orthogonal_centroids}_batch.txt"
        ), "a")
    else:
        fname = open(os.path.join(
            output_dir,
            f"signals_to_probe_inference_under_sampling_exponent={under_sampling_exponent}_var1={var1}_var2={var2}_ortho={orthogonal_centroids}_batch.txt"
        ), "a")

    print(f"\n{'=' * 80}")
    print(f"GEP Gaussian Mixture — probing the inference of the class centroids (BATCH PROCESSING + EMPIRICAL MEANS)")
    print(f"{'=' * 80}")
    print(f"Parameters: d={d}, p={p}, n_samples={n_samples}, n_realizations={n_realizations}, activation={activation}")
    print(f"Batch size: {batch_size}, weight: {weight1}, variance cluster 1: {var1}, variance cluster 2: {var2}")
    print(f"{'=' * 80}\n")

    for realization_idx in range(n_realizations):
        print(f"\n[Realization {realization_idx + 1}/{n_realizations}]")

        # Sweep the diffusion time from t_ini down to t_fin.
        for count, t in enumerate(t_diffusion):
            print(f"\n Time step {count + 1}/{n_times}: t={t:.4f}, "
                  f"adiff={adiff[count]:.4f}, Delta_t={Delta_t[count]:.4f}")

            # Fresh dataset and random features for this (realization, time) pair.
            X, centroid1, centroid2 = generate_dataset(n_samples, d, weight1, [var1, var2],
                                                        centroids_factor1, centroids_factor2, r=0.0)
            W = rng.standard_normal((p, d))

            # ---- Precompute cluster-specific quantities (only 2 clusters) ----
            n_cluster1 = int(n_samples * weight1)
            n_cluster2 = n_samples - n_cluster1

            # Cluster label per sample: 0 for cluster 1, 1 for cluster 2.
            cluster_indices = np.array([0 if i < n_cluster1 else 1 for i in range(n_samples)])

            # Projected centroids: mu_lambda = W @ centroid / sqrt(d).
            mu_lambda_cluster1 = np.dot(W, centroid1) / np.sqrt(d)
            mu_lambda_cluster2 = np.dot(W, centroid2) / np.sqrt(d)

            # Gamma_t (noised std) for each cluster.
            Gamma_t1 = np.sqrt(var1 * adiff[count] ** 2 + Delta_t[count])
            Gamma_t2 = np.sqrt(var2 * adiff[count] ** 2 + Delta_t[count])

            # Precompute the Hermite integrals for both clusters (key optimization).
            print(f"  Precomputing Hermite integrals for 2 clusters...")
            hermite_integrals_cluster1 = compute_hermite_integrals_GEP_hybrid(
                phi, Gamma_t1, mu_lambda_cluster1, var1, adiff[count], Delta_t[count], p, xi, gaussian_weights
            )
            hermite_integrals_cluster2 = compute_hermite_integrals_GEP_hybrid(
                phi, Gamma_t2, mu_lambda_cluster2, var2, adiff[count], Delta_t[count], p, xi, gaussian_weights
            )
            print(f"  ✓ Hermite integrals precomputed!")
            # (Optional) dump the coefficients — uncomment to enable:
            # print_hermite_integrals(hermite_integrals_cluster1, os.path.join(output_dir, "hermite_integrals_cluster1.txt"), p)
            # print_hermite_integrals(hermite_integrals_cluster2, os.path.join(output_dir, "hermite_integrals_cluster2.txt"), p)

            # ---- Assemble U from per-sample contributions, in parallel batches ----
            n_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
            batches = [
                list(range(i * batch_size, min((i + 1) * batch_size, n_samples)))
                for i in range(n_batches)
            ]

            print(f"  Processing {n_samples} samples in {n_batches} batches (batch_size={batch_size})...")
            checkpoint_interval = max(1, batch_size // 5)  # ~5 progress prints per batch
            batch_results = Parallel()(
                delayed(process_batch)(
                    batch_indices, X, W, cluster_indices,
                    hermite_integrals_cluster1, hermite_integrals_cluster2,
                    mu_lambda_cluster1, mu_lambda_cluster2,
                    var1, var2, Gamma_t1, Gamma_t2,
                    adiff[count], Delta_t[count], p, n_samples, d,
                    batch_id=batch_idx + 1, checkpoint_interval=checkpoint_interval
                ) for batch_idx, batch_indices in enumerate(batches)
            )

            # Sum all batch contributions into the full empirical U.
            U_GEP = np.sum(np.array(batch_results), axis=0)
            print(f"  ✓ Computed U matrix ({p}×{p}) at time t={t} with diffusion type {diffusion_type}.")

            # ---- Diagonalize U and probe the inference of the centroids ----
            print("  Computing eigenvalues...")
            eigs_GEP, eigvecs_GEP = np.linalg.eigh(U_GEP)  # ascending eigenvalues
            print(f"  ✓ Eigenvalues computed.")
            print(f"  ✓ Range of U spectrum: [{eigs_GEP.min():.6f}, {eigs_GEP.max():.6f}]")

            # Cosine similarity of the top two eigenvectors with each projected centroid.
            # top_eigvecs[:, 1] is v1 (largest eigenvalue), top_eigvecs[:, 0] is v2.
            top_eigvecs = eigvecs_GEP[:, -2:]

            cos_sim_eigvec2_cluster1 = np.dot(top_eigvecs[:, 0], mu_lambda_cluster1) / (np.linalg.norm(top_eigvecs[:, 0]) * np.linalg.norm(mu_lambda_cluster1))
            cos_sim_eigvec2_cluster2 = np.dot(top_eigvecs[:, 0], mu_lambda_cluster2) / (np.linalg.norm(top_eigvecs[:, 0]) * np.linalg.norm(mu_lambda_cluster2))
            cos_sim_eigvec1_cluster1 = np.dot(top_eigvecs[:, 1], mu_lambda_cluster1) / (np.linalg.norm(top_eigvecs[:, 1]) * np.linalg.norm(mu_lambda_cluster1))
            cos_sim_eigvec1_cluster2 = np.dot(top_eigvecs[:, 1], mu_lambda_cluster2) / (np.linalg.norm(top_eigvecs[:, 1]) * np.linalg.norm(mu_lambda_cluster2))

            # Average cosine similarity of all remaining eigenvectors with each centroid.
            other_eigvecs = eigvecs_GEP[:, :-2]
            cos_sim_other_eigvecs_cluster1_avg = np.mean(other_eigvecs.T @ mu_lambda_cluster1 / (np.linalg.norm(other_eigvecs, axis=0) * np.linalg.norm(mu_lambda_cluster1)))
            cos_sim_other_eigvecs_cluster2_avg = np.mean(other_eigvecs.T @ mu_lambda_cluster2 / (np.linalg.norm(other_eigvecs, axis=0) * np.linalg.norm(mu_lambda_cluster2)))

            # Append one row of inference signals for this diffusion time.
            print("Saving cosine similarities to file...")
            print(t, eigs_GEP[-1], eigs_GEP[-2], eigs_GEP[-3],
                  cos_sim_eigvec1_cluster1, cos_sim_eigvec1_cluster2,
                  cos_sim_eigvec2_cluster2, cos_sim_eigvec2_cluster1,
                  cos_sim_other_eigvecs_cluster1_avg, cos_sim_other_eigvecs_cluster2_avg,
                  file=fname)

    fname.close()
    print(f"\n✅ Done. Results saved under: {output_dir}")
    print(f"Total runtime: {time.time() - start_time:.1f} s")
