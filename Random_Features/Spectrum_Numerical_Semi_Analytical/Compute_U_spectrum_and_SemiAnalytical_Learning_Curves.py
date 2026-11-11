"""
================================================================================
 Spectral analysis of a diffusion-model feature matrix for a Gaussian mixture
================================================================================

OUTLINE OF THE SCRIPT
---------------------
We study a two-cluster Gaussian mixture (a simple "imbalanced classes" model)
and a random-feature representation of it, of the kind that appears in the
analysis of diffusion generative models. For several random realizations we:

  1. Draw a synthetic dataset of `n_samples` points in dimension `d`, split into
     two Gaussian clusters with tunable weights, variances and centroid geometry.
  2. Build a random projection matrix W (the "random features", shape p x d).
  3. Compute a set of coefficients ("Hermite integrals":
     alpha, beta, beta_tilde, gamma_tilde) that describe how the activation
     function (tanh by default; chosen with -act from the ACTIVATIONS dictionary)
     acts on Gaussian inputs. For mixtures with non-zero centroids, the coefficients are different across different dimensions of hidden features space, and are collected in a p x 4 array; for mixtures with zero centroids, the coefficients are identical across dimensions and can be stored in a single 4-vector.
     Coefficients are computed for each cluster of the mixture with fast 1D/2D
     trapezoidal numerical integration.
  4. Assemble two p x p matrices per realization:
        - U        : the empirical feature covariance (sample-based),
        - U_tilde  : the "population"/generalization counterpart (sample-free).
  5. Diagonalize U and U_tilde, then compute the semi-analytical learning
     curves (training loss, test loss, score-matching loss, see README file for definitions) as functions of
     training time.
  6. Write all spectral quantities and loss curves to text files under `results_spectrum/`.

The sample loop is split into BATCHES that run in parallel (joblib),
and the integrals involved in the evaluation of the coefficients are JIT-compiled with numba for speed.

--------------------------------------------------------------------------------
HOW TO RUN THE SCRIPT
--------------------------------------------------------------------------------
Basic run with defaults:

    python GEP_gaussian_mixture_UDiff_batch_userfriendly.py

Custom run (dimension d=100, p=800 features, 400 samples, 10 realizations,
batch size 50):

    python GEP_gaussian_mixture_UDiff_batch_userfriendly.py -d 100 -p 800 -s 400 -r 10 -b 50

Same run with ReLU activation:

    python GEP_gaussian_mixture_UDiff_batch_userfriendly.py -d 100 -p 800 -s 400 -r 10 -b 50 -act relu

See all options:

    python GEP_gaussian_mixture_UDiff_batch_userfriendly.py --help

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    numpy, numba, joblib, matplotlib  (matplotlib only needed if you re-enable
    the optional plotting blocks at the end).

--------------------------------------------------------------------------------
STRUCTURE OF THE OUTPUT FOLDER
--------------------------------------------------------------------------------
A nested folder is created under `results_spectrum/` whose name encodes the main
parameters (including the activation). Inside you will find, per parameter set:
    - spectral_quantities_*.txt           (eigenvalues, IPRs, nu diagonal)
    - spectral_quantities_cluster1/2_*.txt (same, restricted to each cluster)
    - hermite_integrals_cluster1/2_*.txt   (Hermite coefficients)
    - analytical_loss_curves_*.txt         (train/test/score losses vs time)
    - omegas_*.txt                         (overlaps between projected eigenvectors of different clusters)
================================================================================
"""

import os
import time
import secrets
import argparse

import numpy as np
import matplotlib.pyplot as plt              # only used by the optional plots at the end
from joblib import Parallel, delayed
from numba import njit, prange

# Record wall-clock start so we can report the total runtime at the end.
start_time = time.time()


# =============================================================================
# 1. COMMAND-LINE ARGUMENTS
# -----------------------------------------------------------------------------
# Every knob of the experiment is exposed here so you can sweep parameters
# from the shell without editing the file. Run with --help to see them all.
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
    description="Spectral analysis of a diffusion feature matrix for a Gaussian mixture "
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
parser.add_argument("-r", "--realizations", type=int, default=1,
                    help="How many independent random realizations to average over.")

# --- Centroid geometry (relative orientation of the two cluster means) ------
parser.add_argument("--ortho", action="store_true",
                    help="Use orthogonal centroids (cosine = 0). Ignored if -cc is given.")
parser.add_argument("-cc", "--centroid_cosine", type=float, default=None,
                    help="Cosine similarity between the two centroids, in [-1, 1]. "
                         "Overrides --ortho when provided.")

# --- Parallelism ------------------------------------------------------------
parser.add_argument("-b", "--batch_size", type=int, default=50,
                    help="Number of samples processed together in one parallel batch.")
parser.add_argument("-j", "--jobs", type=int, default=-1,
                    help="Number of parallel worker processes (-1 = use all CPU cores).")

# --- Gaussian mixture parameters -------------------------------------------
parser.add_argument("-w", "--weight", type=float, default=0.5,
                    help="Mixing weight of cluster 1 (cluster 2 gets 1 - weight).")
parser.add_argument("-var1", "--var1", type=float, default=0.5,
                    help="Variance of cluster 1.")
parser.add_argument("-var2", "--var2", type=float, default=0.5,
                    help="Variance of cluster 2.")
parser.add_argument("-c1", "--centroid_factor1", type=float, default=3.0,
                    help="Scale (length) factor of centroid 1.")
parser.add_argument("-c2", "--centroid_factor2", type=float, default=3.0,
                    help="Scale (length) factor of centroid 2.")

# --- Activation function ----------------------------------------------------
parser.add_argument("-act", "--activation", type=str.lower, default="tanh",
                    choices=list(ACTIVATIONS),
                    help="Activation function of the random features (keys of ACTIVATIONS).")

args = parser.parse_args()

# Copy parsed arguments into plain variables for readability further down.
d = args.dimension
p = args.projections
n_samples = args.samples
n_realizations = args.realizations
orthogonal_centroids = args.ortho
centroid_cosine_similarity = args.centroid_cosine
batch_size = args.batch_size
n_jobs = args.jobs
weight = args.weight
var1 = args.var1
var2 = args.var2
centroids_factor1 = args.centroid_factor1
centroids_factor2 = args.centroid_factor2
activation = args.activation

# JIT-compile the chosen activation so that the numba integrators below can call it.
phi = njit(ACTIVATIONS[activation])

# Centroid construction mode:
#   True  -> homogeneous/quantized: m1 = c1 * ones,  m2 = c2 * sign_vector
#            (cosine is achieved by choosing how many +1's are in the sign vector)
#   False -> Gram-Schmidt: exact cosine but non-homogeneous components.
homogeneous_centroids = True


# =============================================================================
# 2. RESOLVE CENTROID GEOMETRY
# -----------------------------------------------------------------------------
# Decide the cosine similarity between the two cluster means from the flags. When this parameter is not specified, default is anti-parallel centroid if --ortho is not set, and orthogonal centroid if --ortho is set.
# =============================================================================
if centroid_cosine_similarity is None:
    # No explicit cosine given: orthogonal -> 0.0, otherwise anti-parallel -> -1.0.
    centroid_cosine_similarity = 0.0 if orthogonal_centroids else -1.0
elif not (-1.0 <= centroid_cosine_similarity <= 1.0):
    raise ValueError("centroid_cosine must lie in [-1, 1]")

# Treat "cosine ~ 0" as the orthogonal case (affects only output folder naming).
orthogonal_centroids = np.isclose(centroid_cosine_similarity, 0.0)
geometry_tag = f"cos={centroid_cosine_similarity:.3f}"   # used only in some filenames


# =============================================================================
# 3. FIXED PHYSICAL / DIFFUSION PARAMETERS
# -----------------------------------------------------------------------------
# These describe the (single) diffusion time step at which we probe the model.
# =============================================================================
weight1 = weight              # weight of cluster 1
weight2 = 1.0 - weight1       # weight of cluster 2

t_diffusion = 0.001           # diffusion time at which the analysis is performed
diffusion_type = "OU"         # "FD" = free diffusion, "OU" = Ornstein-Uhlenbeck

# Number of samples in each cluster (cluster 1 first, then cluster 2).
n1 = int(n_samples * weight1)
n2 = n_samples - n1

# Reproducibility note: we seed from OS entropy, so each run differs. To make a
# run reproducible, replace the next line with a fixed integer seed. To reproduce a past run, look in the log file for the seed that was used and set it here.
seed = secrets.randbits(128)
rng = np.random.default_rng(seed)

# Echo the configuration so the log file records exactly what was run.
print(
    f"Parameters: d={d}, p={p}, n_samples={n_samples}, n_realizations={n_realizations}, "
    f"centroid_cosine={centroid_cosine_similarity:.3f}, activation={activation}"
)
print(f"Batch size: {batch_size}, Parallel jobs: {n_jobs}, weight: {weight}, "
      f"variance cluster 1: {var1}, variance cluster 2: {var2}")
print("seed =", seed)

# Translate the diffusion type into the two scalars that enter every formula:
#   adiff   : multiplicative decay applied to the clean signal
#   Delta_t : variance of the injected noise
if diffusion_type == "FD":            # Free diffusion: signal kept, noise grows linearly.
    adiff = 1.0
    Delta_t = t_diffusion
elif diffusion_type == "OU":          # Ornstein-Uhlenbeck: signal decays, noise saturates.
    adiff = np.exp(-t_diffusion)
    Delta_t = 1.0 - np.exp(-2.0 * t_diffusion)

lr = 5e-4                             # learning rate used in the analytical loss curves


# =============================================================================
# 4. PREALLOCATE OUTPUT ARRAYS
# -----------------------------------------------------------------------------
# One row per realization; columns are the p eigenvalues.
#   *_GEP      : spectrum of the empirical matrix U (and per cluster),
#   *_tilde_*  : spectrum of the population matrix U_tilde (and per cluster).
# =============================================================================
all_eigs_GEP          = np.zeros((n_realizations, p))
all_eigs_GEP_cluster1 = np.zeros((n_realizations, p))
all_eigs_GEP_cluster2 = np.zeros((n_realizations, p))
all_eigs_tilde_GEP      = np.zeros((n_realizations, p))
all_eigs_tilde_cluster1 = np.zeros((n_realizations, p))
all_eigs_tilde_cluster2 = np.zeros((n_realizations, p))


# =============================================================================
# 5. INTEGRATION GRID
# -----------------------------------------------------------------------------
# The Hermite integrals are expectations over standard Gaussian variables.
# We approximate them with the trapezoidal rule on a fixed, dense grid.
# `gaussian_weights` is the (unnormalized) Gaussian density exp(-x^2/2);
# the 1/sqrt(2*pi) normalization is applied inside the integrands.
# =============================================================================
n_grid = 2000                                   # grid resolution
xi_max = 10.0                                   # truncate the Gaussian tails at +/- 10 sigma
xi = np.linspace(-xi_max, xi_max, n_grid)       # the integration nodes
gaussian_weights = np.exp(-xi**2 / 2.0)         # exp(-x^2/2), reused everywhere

print(f"Integration grid: {n_grid} points on [-{xi_max}, {xi_max}]")
print("Using fast 1D/2D trapezoidal integration for the Hermite coefficients.")


# =============================================================================
# 6. CORE NUMERICAL ROUTINES
# =============================================================================

def generate_dataset(n, d, weight_cluster_1, vars, centroids_factor_1,
                      centroids_factor_2, centroid_cosine, homogeneous=True):
    """Draw a 2-cluster Gaussian-mixture dataset.

    Parameters
    ----------
    n : int                  total number of points.
    d : int                  ambient dimension.
    weight_cluster_1 : float fraction of points belonging to cluster 1.
    vars : (float, float)    (variance of cluster 1, variance of cluster 2).
    centroids_factor_1/2 : float   length scale of each centroid.
    centroid_cosine : float  desired cosine similarity between the two means.
    homogeneous : bool       centroid construction mode (see top of file).

    Returns
    -------
    [dataset, m_1, m_2] : the (n, d) data matrix and the two centroid vectors.
    """
    n_1 = int(weight_cluster_1 * n)
    n_2 = n - n_1

    var_1, var_2 = vars[0], vars[1]

    # Centroid 1 always points along the all-ones direction.
    m_1 = centroids_factor_1 * np.ones(d)

    if homogeneous:
        # ---- Homogeneous / quantized construction -------------------------
        # m_2 = c2 * sign_vec, where sign_vec has its first k entries +1 and
        # the rest -1. Then cosine(m_1, m_2) = (2k - d)/d, so we invert to get k. Note that this construction is only approximate: the cosine is quantized to the nearest achievable value.
        k = int(round(d * (1.0 + centroid_cosine) / 2.0))
        sign_vec = np.ones(d)
        sign_vec[k:] = -1.0
        m_2 = centroids_factor_2 * sign_vec
    else:
        # ---- Smooth Gram-Schmidt construction -----------------------------
        # Build m_2 with EXACTLY the requested cosine by projecting out the
        # component of a reference axis along m_1 and recombining.
        m1_norm = np.linalg.norm(m_1)
        m2_norm = centroids_factor_2 * np.sqrt(d)
        if d == 1:
            if not np.isclose(abs(centroid_cosine), 1.0):
                raise ValueError("For d=1, centroid_cosine must be either -1 or 1")
            m_2 = np.array([np.sign(centroid_cosine) * centroids_factor_2])
        else:
            u = m_1 / m1_norm                     # unit vector along m_1
            e1 = np.zeros(d); e1[0] = 1.0
            v = e1 - np.dot(e1, u) * u             # component of e1 orthogonal to u
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:                     # e1 was (anti)parallel to u: use e2
                e2 = np.zeros(d); e2[1] = 1.0
                v = e2 - np.dot(e2, u) * u
                v_norm = np.linalg.norm(v)
            v = v / v_norm
            # m_2 = |m_2| * (cos * u + sin * v)
            m_2 = m2_norm * (centroid_cosine * u + np.sqrt(1.0 - centroid_cosine**2) * v)
        

    # Draw samples for the two clusters: isotropic Gaussians centered at m_1 and m_2.
    cluster_1_samples = rng.standard_normal((n_1, d)) * np.sqrt(var_1) + m_1
    cluster_2_samples = rng.standard_normal((n_2, d)) * np.sqrt(var_2) + m_2

    # Stack them (handle the degenerate cases where one cluster is empty).
    if n_1 == 0:
        dataset = cluster_2_samples
    elif n_2 == 0:
        dataset = cluster_1_samples
    else:
        dataset = np.vstack([cluster_1_samples, cluster_2_samples])

    return [dataset, m_1, m_2]


@njit(cache=True)
def trapz_1d(y, x):
    """Numba-friendly trapezoidal rule for 1D arrays (a stand-in for np.trapz)."""
    n = y.shape[0]
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s


# No cache=True: the activation phi is passed as an argument, and numba cannot
# reliably cache functions that take other compiled functions as arguments
# (compilation takes ~1 s per run).
@njit(parallel=True)
def compute_beta_trapz_2d(phi, mu, var_sample, adiff, Delta_t, p, xi, gaussian_weights):
    """Compute the 'beta' coefficient via 2D trapezoidal integration.

    beta[i] models a correlated double expectation of the activation phi
    (a numba-compiled function, see ACTIVATIONS):

        beta[i] = INT p(z1) * [ INT phi(scale1*z1 + scale2*z2 + shift_i) p(z2) dz2 ]^2 dz1

    Returns
    -------
    beta_vals : (p,) array of beta coefficients.
    """
    beta_vals = np.zeros(p)
    n_grid = len(xi)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    scale_1 = np.sqrt(var_sample) * adiff       # weight of the "signal" Gaussian z1
    scale_2 = np.sqrt(Delta_t)                  # weight of the "noise"  Gaussian z2

    # Normalized Gaussian density on the grid.
    gaussian_pdf = gaussian_weights * inv_sqrt_2pi

    for i in prange(p):                          # parallel over features
        shift_i = adiff * mu[i]                  # per-feature mean shift
        integrand_xi1 = np.zeros(n_grid)

        # ----- Outer integral over z1 -----
        for idx1 in range(n_grid):
            # ----- Inner integral over z2 -----
            integral_xi2_vals = np.zeros(n_grid)
            for idx2 in range(n_grid):
                noise = scale_1 * xi[idx1] + scale_2 * xi[idx2] + shift_i
                integral_xi2_vals[idx2] = phi(noise) * gaussian_pdf[idx2]

            # Trapezoidal integral over z2.
            sum_xi2 = 0.0
            for j in range(n_grid - 1):
                sum_xi2 += 0.5 * (integral_xi2_vals[j] + integral_xi2_vals[j + 1]) * (xi[j + 1] - xi[j])

            # Square the inner result and weight by p(z1).
            integrand_xi1[idx1] = (sum_xi2 ** 2) * gaussian_pdf[idx1]

        # Trapezoidal integral over z1.
        sum_xi1 = 0.0
        for i1 in range(n_grid - 1):
            sum_xi1 += 0.5 * (integrand_xi1[i1] + integrand_xi1[i1 + 1]) * (xi[i1 + 1] - xi[i1])

        beta_vals[i] = sum_xi1

    return beta_vals


# No cache=True: the activation phi is passed as an argument, and numba cannot
# reliably cache functions that take other compiled functions as arguments
# (compilation takes ~1 s per run).
@njit(parallel=True)
def compute_hermite_integrals_GEP_hybrid(phi, Gamma_t, mu, var_sample, adiff, Delta_t,
                                         p, xi, gaussian_weights):
    """Compute all four Hermite coefficients for every feature.

    For the activation phi (a numba-compiled function, see ACTIVATIONS) and for
    each feature i (with z a standard Gaussian) we return:
        alpha[i]       = E[ phi(Gamma_t*z + adiff*mu[i]) ]
        beta_tilde[i]  = E[ phi^2(Gamma_t*z + adiff*mu[i]) ]
        gamma_tilde[i] = E[ phi(Gamma_t*z + adiff*mu[i]) * z ]
        beta[i]        = correlated double expectation (see compute_beta_trapz_2d)

    alpha vanishes for odd activations with zero centroids (e.g. tanh), but not
    in general (e.g. ReLU).

    The first three are cheap 1D integrals; the last is the 2D integral above.

    Returns
    -------
    hermite_integrals : (p, 4) array, columns = [alpha, beta_tilde, gamma_tilde, beta].
    """
    hermite_integrals = np.zeros((p, 4))
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    # ---- 1D integrals: alpha, beta_tilde, gamma_tilde (parallel over features) ----
    for i in prange(p):
        shift_i = adiff * mu[i]
        phi_vals = phi(Gamma_t * xi + shift_i)        # activation on the whole grid at once

        # alpha = E[phi(...)]
        integrand_alpha = phi_vals * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 0] = trapz_1d(integrand_alpha, xi)

        # beta_tilde = E[phi^2(...)]
        integrand_beta_tilde = phi_vals**2 * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 1] = trapz_1d(integrand_beta_tilde, xi)

        # gamma_tilde = E[phi(...) * z]
        integrand_gamma_tilde = phi_vals * xi * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 2] = trapz_1d(integrand_gamma_tilde, xi)

    print("  Computed alpha, beta_tilde, gamma_tilde (1D trapz).")

    # ---- 2D integral: beta ----
    hermite_integrals[:, 3] = compute_beta_trapz_2d(
        phi, mu, var_sample, adiff, Delta_t, p, xi, gaussian_weights
    )
    print("  Computed beta (2D trapz).")

    return hermite_integrals


# =============================================================================
# 7. FILE-WRITING HELPERS
# =============================================================================

def print_hermite_integrals(hermite_integrals, fname, p):
    """Append the (p x 4) Hermite coefficients to a text file."""
    with open(fname, "a") as f:
        print("# Index\talpha\tbeta_tilde\tgamma_tilde\tbeta", file=f)
        for i in range(p):
            print(i,
                  hermite_integrals[i, 0], hermite_integrals[i, 1],
                  hermite_integrals[i, 2], hermite_integrals[i, 3], file=f)


def print_spectral_quantities(eigs, eigvecs, eigs_tilde, eigvecs_tilde, nu, fname, p):
    """Append per-eigenvalue spectral quantities to an (already open) file.

    Columns written: index, eigenvalue(U), IPR(U), eigenvalue(U_tilde),
    IPR(U_tilde), nu_diagonal. The IPR (inverse participation ratio) measures
    how localized each eigenvector is.
    """
    iprs = np.sum(eigvecs**4, axis=0)
    iprs_tilde = np.sum(eigvecs_tilde**4, axis=0)
    for i in range(p):
        print(i, eigs[i], iprs[i], eigs_tilde[i], iprs_tilde[i], nu[i], file=fname)


def print_omegas(random_features, eigvecs, gamma_tildes, fname, p, d):
    """Compute and save the per-cluster 'omega' overlaps for each eigenvector.

    omega_c[:, i] projects eigenvector i back into data space, weighted by the
    cluster-c gamma_tilde coefficients. We store ||omega1||^2, ||omega2||^2 and
    their dot product per eigenvector.
    """
    gamma_tilde1, gamma_tilde2 = gamma_tildes[0], gamma_tildes[1]

    omega1 = (np.outer(np.ones(d), gamma_tilde1) * random_features.T @ eigvecs) / np.sqrt(d)
    omega2 = (np.outer(np.ones(d), gamma_tilde2) * random_features.T @ eigvecs) / np.sqrt(d)

    with open(fname, "w") as f_omegas:
        for i in range(p):
            print(i,
                  np.linalg.norm(omega1[:, i])**2,
                  np.linalg.norm(omega2[:, i])**2,
                  np.dot(omega1[:, i], omega2[:, i]), file=f_omegas)


# =============================================================================
# 8. PROCESS A BATCH OF SAMPLES  (runs in parallel, one call per batch)
# =============================================================================

def process_batch(batch_indices, X, W, cluster_indices,
                  hermite_integrals_cluster1, hermite_integrals_cluster2,
                  mu_lambda_cluster1, mu_lambda_cluster2,
                  var1, var2, Gamma_t1, Gamma_t2,
                  adiff, Delta_t, p, n_samples, d_dim,
                  batch_id=None, checkpoint_interval=10):
    """Accumulate the U-matrix contribution of a subset (batch) of samples.

    Returns
    -------
    (U_GEP_batch, U_cluster1_batch, U_cluster2_batch)
        Summed p x p contributions for: all samples in the batch, the cluster-1
        samples only, and the cluster-2 samples only.
    """
    U_GEP_batch      = np.zeros((p, p))
    U_cluster1_batch = np.zeros((p, p))
    U_cluster2_batch = np.zeros((p, p))
    batch_size_actual = len(batch_indices)

    # Each worker gets its own RNG to avoid contention / correlated noise.
    rng = np.random.default_rng(secrets.randbits(128))

    # W @ W.T is the same for every sample, so compute it once.
    WWT = np.dot(W, W.T)

    # gamma coefficients derived from the precomputed gamma_tilde of each cluster.
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

        # Pick the Hermite coefficients for this sample's cluster.
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

        # Lambda = projection of this sample through the random features.
        Lambda = np.dot(W, X_n) / np.sqrt(d_dim)

        alpha      = hermite_integrals[:, 0]
        beta_tilde = hermite_integrals[:, 1]
        beta       = hermite_integrals[:, 3]

        # Gaussian noise originating from the Gaussian Equivalence Principle (GEP) approximation.
        eta = rng.standard_normal(size=p)

        # The variance term must stay non-negative (clip tiny negatives from roundoff).
        variance_term = np.maximum(beta - gamma**2 - alpha**2, 0.0)

        # GEP feature vector for this sample.
        GEP_vector = (alpha
                      + gamma * (Lambda - mu_lambda) / np.sqrt(var_sample)
                      + np.sqrt(variance_term) * eta)

        # Outer product -> rank-1 contribution to the covariance, normalized by n.
        U_GEP_sample = np.outer(GEP_vector, GEP_vector) / n_samples

        # t-dependent W @ W.T term (shared structure across features).
        U_GEP_sample += Delta_t / (var_sample * adiff**2 * d_dim) * WWT * np.outer(gamma, gamma) / n_samples

        # Diagonal
        diag_correction = (beta_tilde - beta - Delta_t / (var_sample * adiff**2) * gamma**2) / n_samples
        U_GEP_sample += np.diag(diag_correction)

        # Accumulate into the total and the per-cluster matrices.
        U_GEP_batch += U_GEP_sample
        if cluster_idx == 0:
            U_cluster1_batch += U_GEP_sample
        else:
            U_cluster2_batch += U_GEP_sample

        # Progress checkpoints within the batch.
        samples_processed += 1
        if batch_id is not None and (samples_processed % checkpoint_interval == 0
                                     or samples_processed == batch_size_actual):
            print(f"    [Batch {batch_id}] Processed {samples_processed}/{batch_size_actual} "
                  f"samples ({100 * samples_processed / batch_size_actual:.0f}%)")

    if batch_id is not None:
        print(f"    [Batch {batch_id}] Complete")

    return U_GEP_batch, U_cluster1_batch, U_cluster2_batch


# =============================================================================
# 9. POPULATION (SAMPLE-FREE) MATRIX U_tilde
# -----------------------------------------------------------------------------
# This is the "generalization" counterpart of U: it needs only the random
# features W and the Hermite coefficients, no data samples.
# =============================================================================
def evaluate_U_tilde(random_features, hermite_coefficients1, hermite_coefficients2):
    """Build U_tilde and its two per-cluster components.

    Returns
    -------
    (Utilde, Utilde1, Utilde2) : the weighted total and the two cluster matrices.
    """
    # Unpack coefficients for both clusters.
    alpha1, beta_tilde1, gamma_tilde1 = (hermite_coefficients1[:, 0],
                                         hermite_coefficients1[:, 1],
                                         hermite_coefficients1[:, 2])
    alpha2, beta_tilde2, gamma_tilde2 = (hermite_coefficients2[:, 0],
                                         hermite_coefficients2[:, 1],
                                         hermite_coefficients2[:, 2])

    # Wishart-like (data-covariance) part, modulated by gamma_tilde.
    gram = random_features @ random_features.T / random_features.shape[1]
    wishartlike_1 = gram * np.outer(gamma_tilde1, gamma_tilde1)
    wishartlike_2 = gram * np.outer(gamma_tilde2, gamma_tilde2)

    # Diagonal (per-feature variance) part.
    diagonal1 = beta_tilde1 - gamma_tilde1**2 - alpha1**2
    diagonal2 = beta_tilde2 - gamma_tilde2**2 - alpha2**2

    # Rank-1 centroid contribution.
    centroids_pert1 = np.outer(alpha1, alpha1)
    centroids_pert2 = np.outer(alpha2, alpha2)

    Utilde1 = wishartlike_1 + np.diag(diagonal1) + centroids_pert1
    Utilde2 = wishartlike_2 + np.diag(diagonal2) + centroids_pert2

    # Mixture-weighted total.
    Utilde = weight1 * Utilde1 + weight2 * Utilde2
    return Utilde, Utilde1, Utilde2


# =============================================================================
# 10. SEMI-ANALYTICAL LEARNING CURVES
# -----------------------------------------------------------------------------
# Given the spectrum of U and the overlap quantities (omegas, nus), produce
# training / test / score losses as functions of (effective) training time and
# write them to file. The math follows linear-network learning dynamics.
# =============================================================================
def compute_and_print_analytical_curves_losses(eigs, omegas, nus,
                                               d, Delta_t, Gammas_t,
                                               weight, fname):
    """Integrate the loss dynamics over a logarithmic grid of training times.

    All inputs are precomputed quantities from one realization. The output is a
    multi-column table (one row per training time) written to `fname`.
    """
    # Unpack grouped arguments.
    Gamma1_t, Gamma2_t = Gammas_t
    omegas1, omegas2 = omegas
    nu, nu1, nu2 = nus

    # ---- Precompute time-independent overlap combinations -----------------
    omega1_combo = np.dot(omegas1.T, omegas1) / Gamma1_t**2
    omega2_combo = np.dot(omegas2.T, omegas2) / Gamma2_t**2
    omega1_sq = np.diagonal(omega1_combo)
    omega2_sq = np.diagonal(omega2_combo)

    omegas_combo = weight * omegas1 / Gamma1_t + (1 - weight) * omegas2 / Gamma2_t
    omegas_combo_dot = np.dot(omegas_combo.T, omegas_combo)
    omegas_sq_combo = np.diagonal(omegas_combo_dot)

    omegas_combo_dot_with_omega1 = np.diagonal(np.dot(omegas_combo.T, omegas1) / Gamma1_t)
    omegas_combo_dot_with_omega2 = np.diagonal(np.dot(omegas_combo.T, omegas2) / Gamma2_t)

    # "Generalization" terms: nu minus the diagonal eigenvalue part.
    gen_term_filtered          = nu  - np.eye(p) * eigs
    gen_term_cluster1_filtered = nu1 - np.eye(p) * eigs

    # Products that do not depend on time (precomputed for speed).
    gen_omega_combo = gen_term_filtered * omegas_combo_dot
    gen_omega1      = gen_term_cluster1_filtered * omegas_combo_dot
    gen_omega1_for_score = (gen_term_cluster1_filtered + np.eye(p) * eigs) * omegas_combo_dot
    gen_omega2_for_score = ((gen_term_filtered - weight * gen_term_cluster1_filtered) / (1 - weight)
                            + np.eye(p) * eigs) * omegas_combo_dot

    # ---- Build the (logarithmic) grid of training times -------------------
    log_t_max = 10
    n_training_times = 1100
    log_training_times = np.linspace(0, log_t_max, n_training_times)
    training_times = np.concatenate(([0], 10**log_training_times))
    eta = lr * d / Delta_t                              # effective learning rate
    training_times_eff = eta * Delta_t / d * training_times

    print(f"Computing analytical loss curves up to t={training_times[-1]:.2e} "
          f"(effective t_eff={training_times_eff[-1]:.2e})")

    # t = 0 row: every (normalized) loss starts at 1.
    print(0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, file=fname)

    # Constant prefactors used inside the loop.
    factor = Delta_t / d
    factor_score = 1.0 / d
    eigs_inv = 1.0 / eigs

    for t in training_times_eff[1:]:
        # Exponential "damping" factors that govern how each mode is learned.
        exp_4t = np.exp(-4 * eigs * t)
        dumping_term = 1 - exp_4t
        dumping_term_over_eigs = dumping_term * eigs_inv

        # ---- Training losses ----
        loss_training          = 1 - np.dot(dumping_term_over_eigs, omegas_sq_combo) * factor
        loss_training_cluster1 = 1 - np.dot(dumping_term_over_eigs, omega1_sq) * factor
        loss_training_cluster2 = 1 - np.dot(dumping_term_over_eigs, omega2_sq) * factor

        # ---- Test losses ----
        exp_2t = np.exp(-2 * eigs * t)
        dumping_term_bis = (1 - exp_2t) * eigs_inv
        # Outer product of the per-mode damping factors (i,j coupling).
        dumping_outer_vec = dumping_term_bis[:, None] * dumping_term_bis[None, :]

        term1          = np.sum(dumping_outer_vec * gen_omega_combo) * factor
        term1_cluster1 = np.sum(dumping_outer_vec * gen_omega1) * factor
        term2_cluster1 = 2 * np.dot(dumping_term_bis,
                                    (omegas_combo_dot_with_omega1 - omegas_sq_combo)) * factor

        loss_test          = loss_training + term1
        loss_test_cluster1 = loss_training + term1_cluster1 - term2_cluster1
        loss_test_cluster2 = (loss_test - weight * loss_test_cluster1) / (1 - weight)

        # ---- Score-matching losses ----
        initial_value_11 = 1 / Gamma1_t**2
        initial_value_22 = 1 / Gamma2_t**2

        term1_score_11 = np.sum(gen_omega1_for_score * dumping_outer_vec) * factor_score
        term1_score_22 = np.sum(gen_omega2_for_score * dumping_outer_vec) * factor_score
        term2_score_11 = 2 * factor_score * np.dot(dumping_term_bis, omegas_combo_dot_with_omega1)
        term2_score_22 = 2 * factor_score * np.dot(dumping_term_bis, omegas_combo_dot_with_omega2)

        Epsilon_score_11 = initial_value_11 + term1_score_11 - term2_score_11
        Epsilon_score_22 = initial_value_22 + term1_score_22 - term2_score_22

        loss_score = ((weight * Epsilon_score_11 + (1 - weight) * Epsilon_score_22)
                      / (weight * initial_value_11 + (1 - weight) * initial_value_22))

        # One row: t, training (total/c1/c2), test (total/c1/c2), score, eps11, eps22.
        print(t,
              loss_training, loss_training_cluster1, loss_training_cluster2,
              loss_test, loss_test_cluster1, loss_test_cluster2,
              loss_score, Epsilon_score_11/initial_value_11, Epsilon_score_22/initial_value_22, file=fname)


# =============================================================================
# 11. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    # ---- Build the output directory (its name encodes the key parameters) ----
    _base_dir = (f"results_spectrum/output_spectral_quantities_sampling_bias_"
                 f"t_diff={t_diffusion}_type={diffusion_type}_"
                 f"centroid1={centroids_factor1}_centroid2={centroids_factor2}_"
                 f"activation={activation}/"
                 f"P={p}_D={d}_S={n_samples}/")
    if not orthogonal_centroids:
        output_dir = os.path.join(_base_dir, f"cosine_similarity={centroid_cosine_similarity:.3f}/")
    else:
        output_dir = _base_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- Open the three main output files (append mode) ----
    fname          = open(os.path.join(output_dir, f"spectral_quantities_weight={weight1}_var1={var1}_var2={var2}_batch.txt"), "a")
    fname_cluster1 = open(os.path.join(output_dir, f"spectral_quantities_cluster1_weight={weight1}_var1={var1}_var2={var2}_batch.txt"), "a")
    fname_cluster2 = open(os.path.join(output_dir, f"spectral_quantities_cluster2_weight={weight1}_var1={var1}_var2={var2}_batch.txt"), "a")

    print(f"\n{'='*80}")
    print("Spectral analysis of a diffusion feature matrix")
    print(f"{'='*80}")
    print(f"Parameters: d={d}, p={p}, n_samples={n_samples}, n_realizations={n_realizations}, activation={activation}")
    print(f"Batch size: {batch_size}, weight: {weight1}, var1: {var1}, var2: {var2}")
    print(f"{'='*80}\n")

    # ====================== LOOP OVER REALIZATIONS ======================
    for realization_idx in range(n_realizations):
        print(f"\n[Realization {realization_idx + 1}/{n_realizations}]")

        # (a) Draw the dataset and obtain the two true centroids.
        X, centroid1, centroid2 = generate_dataset(
            n_samples, d, weight1, [var1, var2],
            centroids_factor1, centroids_factor2, centroid_cosine_similarity,
            homogeneous=homogeneous_centroids,
        )

        # (b) Draw the random feature matrix (p x d).
        W = rng.standard_normal((p, d))

        # (c) Label each sample by cluster: 0 for cluster 1, 1 for cluster 2.
        n_cluster1 = int(n_samples * weight1)
        cluster_indices = np.array([0 if i < n_cluster1 else 1 for i in range(n_samples)])

        # (d) Project the centroids through the random features.
        mu_lambda_cluster1 = np.dot(W, centroid1) / np.sqrt(d)
        mu_lambda_cluster2 = np.dot(W, centroid2) / np.sqrt(d)

        # (e) Per-cluster diffusion scale Gamma_t = sqrt(var * adiff^2 + Delta_t).
        Gamma_t1 = np.sqrt(var1 * adiff**2 + Delta_t)
        Gamma_t2 = np.sqrt(var2 * adiff**2 + Delta_t)

        # (f) Precompute the Hermite coefficients for each cluster.
        print("  Precomputing Hermite integrals for the 2 clusters...")
        hermite_integrals_cluster1 = compute_hermite_integrals_GEP_hybrid(
            phi, Gamma_t1, mu_lambda_cluster1, var1, adiff, Delta_t, p, xi, gaussian_weights
        )
        hermite_integrals_cluster2 = compute_hermite_integrals_GEP_hybrid(
            phi, Gamma_t2, mu_lambda_cluster2, var2, adiff, Delta_t, p, xi, gaussian_weights
        )
        print("  Hermite integrals precomputed.")
        print_hermite_integrals(hermite_integrals_cluster1,
                                os.path.join(output_dir, f"hermite_integrals_cluster1_weight={weight1}_var1={var1}_var2={var2}_batch.txt"), p)
        print_hermite_integrals(hermite_integrals_cluster2,
                                os.path.join(output_dir, f"hermite_integrals_cluster2_weight={weight1}_var1={var1}_var2={var2}_batch.txt"), p)

        # (g) Split the samples into batches and process them in parallel.
        n_batches = (n_samples + batch_size - 1) // batch_size   # ceiling division
        batches = [
            list(range(i * batch_size, min((i + 1) * batch_size, n_samples)))
            for i in range(n_batches)
        ]
        print(f"  Processing {n_samples} samples in {n_batches} batches (batch_size={batch_size})...")
        checkpoint_interval = max(1, batch_size // 5)            # ~5 progress prints per batch

        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(
                batch_indices, X, W, cluster_indices,
                hermite_integrals_cluster1, hermite_integrals_cluster2,
                mu_lambda_cluster1, mu_lambda_cluster2,
                var1, var2, Gamma_t1, Gamma_t2,
                adiff, Delta_t, p, n_samples, d,
                batch_id=batch_idx + 1, checkpoint_interval=checkpoint_interval,
            )
            for batch_idx, batch_indices in enumerate(batches)
        )

        # (h) Sum the batch contributions into the full U matrices.
        #     Per-cluster matrices are renormalized by their mixture weight.
        U_GEP      = np.sum(np.array([r[0] for r in batch_results]), axis=0)
        U_cluster1 = np.sum(np.array([r[1] for r in batch_results]), axis=0) / weight1
        U_cluster2 = np.sum(np.array([r[2] for r in batch_results]), axis=0) / weight2
        print(f"  Computed U matrices ({p}x{p}): total, cluster1, cluster2")

        # (i) Build the population matrix U_tilde (no samples needed).
        Utilde, Utilde1, Utilde2 = evaluate_U_tilde(
            W, hermite_integrals_cluster1, hermite_integrals_cluster2
        )
        print(f"  Computed U_tilde matrices ({p}x{p}): total, cluster1, cluster2")

        # (j) Diagonalize every matrix (eigh -> ascending real eigenvalues).
        print("  Computing eigenvalues...")
        eigs_GEP,          eigvecs_GEP          = np.linalg.eigh(U_GEP)
        eigs_cluster1,     eigvecs_cluster1     = np.linalg.eigh(U_cluster1)
        eigs_cluster2,     eigvecs_cluster2     = np.linalg.eigh(U_cluster2)
        eigs_tilde_GEP,      eigvecs_tilde_GEP      = np.linalg.eigh(Utilde)
        eigs_tilde_cluster1, eigvecs_tilde_cluster1 = np.linalg.eigh(Utilde1)
        eigs_tilde_cluster2, eigvecs_tilde_cluster2 = np.linalg.eigh(Utilde2)
        print("  Eigenvalues computed.")
        print(f"    U   spectrum range: [{eigs_GEP.min():.6f}, {eigs_GEP.max():.6f}]")
        print(f"    U1  spectrum range: [{eigs_cluster1.min():.6f}, {eigs_cluster1.max():.6f}]")
        print(f"    U2  spectrum range: [{eigs_cluster2.min():.6f}, {eigs_cluster2.max():.6f}]")
        print(f"    Ut  spectrum range: [{eigs_tilde_GEP.min():.6f}, {eigs_tilde_GEP.max():.6f}]")
        print(f"    Ut1 spectrum range: [{eigs_tilde_cluster1.min():.6f}, {eigs_tilde_cluster1.max():.6f}]")
        print(f"    Ut2 spectrum range: [{eigs_tilde_cluster2.min():.6f}, {eigs_tilde_cluster2.max():.6f}]")

        # (k) "nu" matrices: U_tilde expressed in the eigenbasis of U.
        #     These encode how the population matrix couples the empirical modes.
        nu  = np.dot(eigvecs_GEP.T, np.dot(Utilde,  eigvecs_GEP))
        nu1 = np.dot(eigvecs_GEP.T, np.dot(Utilde1, eigvecs_GEP))
        nu2 = np.dot(eigvecs_GEP.T, np.dot(Utilde2, eigvecs_GEP))

        # (l) Overlap ("omega") quantities needed for the semi-analytical loss curves.
        gamma_tilde1 = hermite_integrals_cluster1[:, 2]
        gamma_tilde2 = hermite_integrals_cluster2[:, 2]
        omega1 = (np.outer(np.ones(d), gamma_tilde1) * W.T @ eigvecs_GEP) / np.sqrt(d)
        omega2 = (np.outer(np.ones(d), gamma_tilde2) * W.T @ eigvecs_GEP) / np.sqrt(d)
        
        # (m) Compute and save the analytical train/test/score loss curves.
        compute_and_print_analytical_curves_losses(
            eigs_GEP,
            [omega1, omega2],
            [nu, nu1, nu2],
            d, Delta_t,
            [Gamma_t1, Gamma_t2],
            weight1,
            open(os.path.join(output_dir, f"analytical_loss_curves_weight={weight1}_var1={var1}_var2={var2}_batch.txt"), "a"),
        )

        # (n) Save all spectral quantities (total + per cluster) and the omegas.
        print_spectral_quantities(eigs_GEP, eigvecs_GEP, eigs_tilde_GEP, eigvecs_tilde_GEP, np.diagonal(nu), fname, p)
        print_spectral_quantities(eigs_cluster1, eigvecs_cluster1, eigs_tilde_cluster1, eigvecs_tilde_cluster1, np.diagonal(nu1), fname_cluster1, p)
        print_spectral_quantities(eigs_cluster2, eigvecs_cluster2, eigs_tilde_cluster2, eigvecs_tilde_cluster2, np.diagonal(nu2), fname_cluster2, p)
        print_omegas(W, eigvecs_GEP,
                     [hermite_integrals_cluster1[:, 2], hermite_integrals_cluster2[:, 2]],
                     os.path.join(output_dir, f"omegas_weight={weight1}_var1={var1}_var2={var2}_{geometry_tag}_batch.txt"), p, d)
        print("  Spectral quantities saved to file.")

        # (o) Keep this realization's eigenvalues for the final summary.
        all_eigs_GEP[realization_idx]          = eigs_GEP
        all_eigs_GEP_cluster1[realization_idx] = eigs_cluster1
        all_eigs_GEP_cluster2[realization_idx] = eigs_cluster2
        all_eigs_tilde_GEP[realization_idx]      = eigs_tilde_GEP
        all_eigs_tilde_cluster1[realization_idx] = eigs_tilde_cluster1
        all_eigs_tilde_cluster2[realization_idx] = eigs_tilde_cluster2

    # ---- Close the main output files ----
    fname.close()
    fname_cluster1.close()
    fname_cluster2.close()

    # ---- Final summary ----
    print(f"\n{'='*80}")
    print("All realizations complete!")
    print(f"{'='*80}")
    print(f"\nTotal execution time: {time.time() - start_time:.2f} s")

    # -------------------------------------------------------------------------
    # OPTIONAL: histogram of the pooled eigenvalue distributions.
    # Uncomment to visualize the spectra of U and U_tilde across realizations.
    # -------------------------------------------------------------------------
    # eigss_GEP      = np.sort(all_eigs_GEP.flatten())
    # eigss_tilde    = np.sort(all_eigs_tilde_GEP.flatten())
    # fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    # axs[0].hist(np.log10(eigss_GEP[eigss_GEP > 0]),
    #             bins=5 * int(np.sqrt(p) + 0.5), density=True, alpha=0.8, edgecolor='black')
    # axs[1].hist(np.log10(eigss_tilde[eigss_tilde > 0]),
    #             bins=5 * int(np.sqrt(p) + 0.5), density=True, alpha=0.8, edgecolor='black')
    # axs[0].set(xlabel="log10(Eigenvalue)", ylabel="Density", title="Spectrum of U")
    # axs[1].set(xlabel="log10(Eigenvalue)", ylabel="Density", title="Spectrum of U_tilde")
    # for ax in axs:
    #     ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
