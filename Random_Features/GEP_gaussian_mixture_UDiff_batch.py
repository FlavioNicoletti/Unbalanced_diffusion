"""
GEP Gaussian Mixture Eigenvalue Computation

Usage:
    python test_GEP_gaussian_mixture_batch.py -d 100 -p 800 -s 400 -r 10 -b 50
    python test_GEP_gaussian_mixture_batch.py --help
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit, prange
import argparse
from math import factorial
import time
import os

from methods_for_GEP_unbalanced_mixture import generate_samples_from_unbalanced_gaussian_mixture, generate_random_weights

start_time = time.time()

# =============================================================================
# Command-line argument parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="GEP Gaussian Mixture Eigenvalue Computation (Batch + Empirical Means)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-d", "--dimension", type=int, default=100,
                    help="Dimension of samples (d)")
parser.add_argument("-p", "--projections", type=int, default=800,
                    help="Number of random projections (p)")
parser.add_argument("-s", "--samples", type=int, default=400,
                    help="Number of samples (n_samples)")
parser.add_argument("-r", "--realizations", type=int, default=10,
                    help="Number of realizations (n_realizations)")
parser.add_argument("--ortho", action="store_true",
                    help="Use orthogonal centroids (default: False)")
parser.add_argument("-b", "--batch_size", type=int, default=50,
                    help="Number of samples per batch (processed in parallel)")
parser.add_argument("-j", "--jobs", type=int, default=-1,
                    help="Number of parallel jobs (-1 = all cores)")
parser.add_argument("-m", "--mc_samples", type=int, default=1000000,
                    help="Number of Monte Carlo samples for empirical mean estimation")
parser.add_argument("-w", "--weight", type=float, default=0.5,
                    help="Weight of first cluster (default w=0.5)")
parser.add_argument("-vr", "--var_ratio", type=float, default=1.0,
                    help="Ratio between the two cluster variances (default vr=1.0)")

args = parser.parse_args()

# Parameters from command line
d = args.dimension
p = args.projections
n_samples = args.samples
n_realizations = args.realizations
orthogonal_centroids = args.ortho
batch_size = args.batch_size
n_jobs = args.jobs
n_mc = args.mc_samples
weight = args.weight
vr = args.var_ratio

# Fixed parameters for the Gaussian mixture
var1 = 0.5
var2 = 0.5*vr
shift = 3.0
weight1 = weight
weight2 = 1 - weight1
t_diffusion = 0.01
diffusion_type = "OU"  # "FD" (Free Diffusion) or "OU" (Ornstein-Uhlenbeck)

print(f"Parameters: d={d}, p={p}, n_samples={n_samples}, n_realizations={n_realizations}, ortho={orthogonal_centroids}")
print(f"Batch size: {batch_size}, Parallel jobs: {n_jobs}, MC samples: {n_mc}, weight: {weight}, variance ratio: {vr}")

if diffusion_type == "FD":
    adiff = 1.0
    Delta_t = t_diffusion
elif diffusion_type == "OU":
    adiff = np.exp(-t_diffusion)
    Delta_t = 1.0 - np.exp(-2.0 * t_diffusion)

# Preallocate array for eigenvalues
all_eigs_GEP = np.zeros((n_realizations, p))
all_eigs_GEP_cluster1 = np.zeros((n_realizations, p))
all_eigs_GEP_cluster2 = np.zeros((n_realizations, p))

# =============================================================================
# Integration grid for trapezoidal rule (alpha, beta_tilde, gamma_tilde)
# =============================================================================
n_grid = 2000  # Number of grid points for integration
xi_max = 10.0  # Integration range: [-xi_max, xi_max]
xi = np.linspace(-xi_max, xi_max, n_grid)
gaussian_weights = np.exp(-xi**2 / 2.0)  # Precompute exp(-x^2/2)

print(f"Integration grid: {n_grid} points on [-{xi_max}, {xi_max}]")

# =============================================================================
# Numba-compatible helper functions
# =============================================================================

@njit(cache=True)
def tanh_vectorized(x):
    """Vectorized tanh for numba"""
    return np.tanh(x)


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


@njit(parallel=True, cache=True)
def compute_hermite_integrals_GEP_hybrid(Gamma_t, mu, var_sample, adiff, p, n_mc, xi, gaussian_weights):
    """
    Compute Hermite integrals for GEP using HYBRID approach:
    - alpha, beta_tilde, gamma_tilde: TRAPEZOIDAL INTEGRATION (fast 1D integrals)
    - beta: MONTE CARLO empirical mean (avoids slow 2D integration)
    
    OPTIMIZED VERSION: Single loop over p dimensions for all 4 coefficients.
    
    For each dimension i, we compute:
    - alpha[i] = E[tanh(Gamma_t * z + adiff * mu[i])]
    - beta_tilde[i] = E[tanh^2(Gamma_t * z + adiff * mu[i])]
    - gamma_tilde[i] = E[tanh(Gamma_t * z + adiff * mu[i]) * z]
    - beta[i] = E[tanh(Gamma_t * z1 + adiff * mu[i]) * tanh(Gamma_t * z2 + adiff * mu[i])]
      where (z1, z2) are correlated with correlation c = adiff^2 * var / Gamma_t^2
    
    Args:
        Gamma_t: diffusion parameter (scalar)
        mu: projected centroid, shape (p,)
        var_sample: variance of the sample's cluster
        adiff: diffusion coefficient
        p: number of projections
        n_mc: number of Monte Carlo samples for beta
        xi: integration grid for trapezoidal rule
        gaussian_weights: exp(-xi^2/2) precomputed weights
    
    Returns:
        hermite_integrals: shape (p, 4) with [alpha, beta_tilde, gamma_tilde, beta]
    """
    hermite_integrals = np.zeros((p, 4))
    
    # Precompute constants
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    
    # Monte Carlo setup for beta (correlated 2D integral)
    c = adiff**2 * var_sample * Gamma_t**(-2)
    sqrt_1_minus_c2 = np.sqrt(1.0 - c**2)
    
    # Generate random samples once (shared across all dimensions)
    z = np.random.randn(n_mc)
    z_indep = np.random.randn(n_mc)
    
    # Correlated pair: z2 = c*z + sqrt(1-c^2)*z_indep
    z2 = c * z + sqrt_1_minus_c2 * z_indep
    
    # Precompute scaled z values for Monte Carlo
    Gamma_z = Gamma_t * z
    Gamma_z2 = Gamma_t * z2
    inv_n_mc = 1.0 / n_mc
    
    # =========================================================================
    # Single parallel loop: compute all 4 coefficients for each dimension
    # =========================================================================
    for i in prange(p):
        shift_i = adiff * mu[i]
        
        # -----------------------------------------------------------------
        # TRAPEZOIDAL: alpha, beta_tilde, gamma_tilde (1D integrals)
        # -----------------------------------------------------------------
        # Precompute tanh values on integration grid
        tanh_vals = np.tanh(Gamma_t * xi + shift_i)
        
        # alpha: integral of tanh(...) * exp(-x^2/2) / sqrt(2*pi)
        integrand_alpha = tanh_vals * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 0] = trapz_1d(integrand_alpha, xi)
        
        # beta_tilde: integral of tanh^2(...) * exp(-x^2/2) / sqrt(2*pi)
        integrand_beta_tilde = tanh_vals**2 * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 1] = trapz_1d(integrand_beta_tilde, xi)
        
        # gamma_tilde: integral of tanh(...) * x * exp(-x^2/2) / sqrt(2*pi)
        integrand_gamma_tilde = tanh_vals * xi * gaussian_weights * inv_sqrt_2pi
        hermite_integrals[i, 2] = trapz_1d(integrand_gamma_tilde, xi)
        
        # -----------------------------------------------------------------
        # MONTE CARLO: beta (2D correlated integral)
        # -----------------------------------------------------------------
        sum_tanh_prod = 0.0
        for k in range(n_mc):
            tanh1 = np.tanh(Gamma_z[k] + shift_i)
            tanh2 = np.tanh(Gamma_z2[k] + shift_i)
            sum_tanh_prod += tanh1 * tanh2
        
        hermite_integrals[i, 3] = sum_tanh_prod * inv_n_mc  # beta
    
    return hermite_integrals


# =============================================================================
# OLD CODE: Numerical integration version (kept for reference)
# =============================================================================
'''
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

@njit(cache=True)
def trapz_axis1(y2d, x):
    """Integrate 2D array along axis=1 (columns) returning 1D array of integrals per row."""
    nrows = y2d.shape[0]
    ncols = y2d.shape[1]
    out = np.zeros(nrows)
    if ncols < 2:
        return out
    for r in range(nrows):
        s = 0.0
        for j in range(ncols - 1):
            s += 0.5 * (y2d[r, j] + y2d[r, j + 1]) * (x[j + 1] - x[j])
        out[r] = s
    return out

@njit(parallel=True, cache=True)
def compute_hermite_integrals_GEP_parallel(xi, Gamma_t, mu, gaussian_weights, gaussian_weights_bivar, corr, p):
    """Compute hermite integrals for all p dimensions in parallel for GEP (NUMERICAL INTEGRATION)"""
    hermite_integrals = np.zeros((p, 4))
    # alpha
    for i in prange(p):
        ph_vals = tanh_vectorized(Gamma_t*xi + adiff*mu[i])
        integrand = ph_vals * gaussian_weights / np.sqrt(2.0 * np.pi)
        hermite_integrals[i, 0] = trapz_1d(integrand, xi)
    # beta tilde
    for i in range(p):
        ph_vals = tanh_vectorized(Gamma_t*xi + adiff*mu[i])
        integrand = ph_vals**2 * gaussian_weights / np.sqrt(2.0 * np.pi)
        hermite_integrals[i, 1] = trapz_1d(integrand, xi)
    # gamma tilde
    for i in range(p):
        ph_vals = tanh_vectorized(Gamma_t*xi + adiff*mu[i])
        integrand = ph_vals * xi * gaussian_weights / np.sqrt(2.0 * np.pi)
        hermite_integrals[i, 2] = trapz_1d(integrand, xi)
    # beta
    for i in range(p):
        ph_vals = tanh_vectorized(Gamma_t*xi + adiff*mu[i])
        integrand1 = corr * np.outer(np.ones(xi.shape), ph_vals*gaussian_weights_bivar)
        integral1 = trapz_axis1(integrand1, xi)
        integrand2 = ph_vals*integral1*gaussian_weights_bivar/(2.0 * np.pi)
        hermite_integrals[i, 3] = trapz_1d(integrand2, xi)
    return hermite_integrals
'''


def process_batch(batch_indices, X, W, cluster_indices, 
                  hermite_integrals_cluster1, hermite_integrals_cluster2,
                  mu_lambda_cluster1, mu_lambda_cluster2,
                  var1, var2, Gamma_t1, Gamma_t2,
                  adiff, Delta_t, p, n_samples, d_dim,
                  batch_id=None, checkpoint_interval=10):
    """
    Process a BATCH of samples and return their summed U_GEP contribution.
    
    OPTIMIZED: Hermite integrals are PRECOMPUTED per cluster (not per sample).
    
    This function is called in parallel for different batches.
    Each batch processes its samples sequentially and returns the sum,
    reducing memory overhead compared to storing all individual matrices.
    
    Args:
        batch_indices: list of sample indices to process in this batch
        X: all samples, shape (n_samples, d)
        W: weight matrix, shape (p, d)
        cluster_indices: cluster index (0 or 1) for each sample, shape (n_samples,)
        hermite_integrals_cluster1: precomputed integrals for cluster 1, shape (p, 4)
        hermite_integrals_cluster2: precomputed integrals for cluster 2, shape (p, 4)
        mu_lambda_cluster1: W @ centroid1 / sqrt(d), shape (p,)
        mu_lambda_cluster2: W @ centroid2 / sqrt(d), shape (p,)
        var1, var2: variances of the two clusters
        Gamma_t1, Gamma_t2: Gamma_t values for the two clusters
        adiff, Delta_t: diffusion parameters
        p: number of projections
        n_samples: total number of samples
        d_dim: dimension d
        batch_id: identifier for this batch (for checkpoint logging)
        checkpoint_interval: print checkpoint every N samples within batch
    
    Returns:
        tuple: (U_GEP_batch, U_cluster1_batch, U_cluster2_batch)
            - U_GEP_batch: sum of U_GEP contributions for all samples in the batch
            - U_cluster1_batch: sum of U_GEP contributions for cluster 1 samples only
            - U_cluster2_batch: sum of U_GEP contributions for cluster 2 samples only
    """
    U_GEP_batch = np.zeros((p, p))
    U_cluster1_batch = np.zeros((p, p))
    U_cluster2_batch = np.zeros((p, p))
    batch_size_actual = len(batch_indices)
    
    # Precompute W @ W.T (used for all samples in batch)
    WWT = np.dot(W, W.T)
    
    # Precompute gamma for both clusters (derived from precomputed hermite integrals)
    gamma_tilde_1 = hermite_integrals_cluster1[:, 2]
    gamma_tilde_2 = hermite_integrals_cluster2[:, 2]
    gamma_cluster1 = gamma_tilde_1 * np.sqrt(var1) * adiff / Gamma_t1
    gamma_cluster2 = gamma_tilde_2 * np.sqrt(var2) * adiff / Gamma_t2
    
    # Checkpoint: batch start
    if batch_id is not None:
        print(f"    [Batch {batch_id}] Starting - {batch_size_actual} samples (indices {batch_indices[0]}-{batch_indices[-1]})")
    
    samples_processed = 0
    for n in batch_indices:
        X_n = X[n, :]
        cluster_idx = cluster_indices[n]
        
        # Select precomputed values based on cluster
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
        
        # Compute Lambda = W @ X_n / sqrt(d)
        Lambda = np.dot(W, X_n) / np.sqrt(d_dim)
        
        # Extract precomputed coefficients
        alpha = hermite_integrals[:, 0]
        beta_tilde = hermite_integrals[:, 1]
        beta = hermite_integrals[:, 3]
        
        # Random noise for GEP
        eta = np.random.normal(size=p)
        
        # Ensure non-negative variance term
        variance_term = np.maximum(beta - gamma**2 - alpha**2, 0.0)
        
        # GEP vector
        GEP_vector = alpha + gamma * (Lambda - mu_lambda) / np.sqrt(var_sample) + np.sqrt(variance_term) * eta
        
        # U_GEP_sample contribution
        U_GEP_sample = np.outer(GEP_vector, GEP_vector) / n_samples
        
        # W @ W.T term
        U_GEP_sample += Delta_t / (var_sample * adiff**2 * d_dim) * WWT * np.outer(gamma, gamma) / n_samples
        
        # Diagonal correction
        diag_correction = (beta_tilde - beta - Delta_t / (var_sample * adiff**2) * gamma**2) / n_samples
        U_GEP_sample += np.diag(diag_correction)
        
        U_GEP_batch += U_GEP_sample
        
        # Also accumulate to the appropriate cluster-specific matrix
        if cluster_idx == 0:
            U_cluster1_batch += U_GEP_sample
        else:
            U_cluster2_batch += U_GEP_sample
        
        # Checkpoint: progress within batch
        samples_processed += 1
        if batch_id is not None and (samples_processed % checkpoint_interval == 0 or samples_processed == batch_size_actual):
            print(f"    [Batch {batch_id}] Processed {samples_processed}/{batch_size_actual} samples ({100*samples_processed/batch_size_actual:.0f}%)")
    
    # Checkpoint: batch complete
    if batch_id is not None:
        print(f"    [Batch {batch_id}] Complete ✓")
    
    return U_GEP_batch, U_cluster1_batch, U_cluster2_batch


def print_spectral_quantities(eigs, eigvecs, proj_centroids, fname, p):
    """Write spectral quantities (eigenvalues and IPRs) to file."""
    iprs = np.sum(eigvecs**4, axis=0)
    c1 = proj_centroids[0]
    c2 = proj_centroids[1]
    for i in range(p):
        o1 = np.dot(eigvecs[:, i], c1) / (np.linalg.norm(eigvecs[:, i]) * np.linalg.norm(c1))
        o2 = np.dot(eigvecs[:, i], c2) / (np.linalg.norm(eigvecs[:, i]) * np.linalg.norm(c2))
        print(i, eigs[i], iprs[i], o1, o2, file=fname)

#@njit(parallel=True, cache=True)
def compute_analytical_curve_training_loss(eigs, eigvecs, eigs1, eigvecs1, eigs2, eigvecs2, d, P, Delta_t, Gammas_t, gamma_tildes, random_features, weight, lr):

    ## compute T matrix

    gamma_tilde1 = gamma_tildes[0]
    gamma_tilde2 = gamma_tildes[1]

    Gamma1_t = Gammas_t[0]
    Gamma2_t = Gammas_t[1]

    g1 = np.outer(gamma_tilde1/Gamma1_t, np.full(d, 1))
    g2 = np.outer(gamma_tilde2/Gamma2_t, np.full(d, 1))

    Wtilde1 = random_features*g1
    Wtilde2 = random_features*g2

    ew = np.linalg.norm((weight*Wtilde1.T+(1-weight)*Wtilde2.T) @ eigvecs, axis=0)**2

    '''
    ew1 = np.linalg.norm(Wtilde1.T @ eigvecs, axis=0)**2
    ew2 = np.linalg.norm(Wtilde2.T @ eigvecs, axis=0)**2

    fname_debug = "debug_analytical_training_loss_curve.txt"
    fdebug = open(fname_debug, "w")
    for i in range(p):
        print(eigs[i], ew1[i]/d, ew2[i]/d, file=fdebug)
    fdebug.close()
    '''

    ew1_cluster1 = np.linalg.norm(Wtilde1.T @ eigvecs1, axis=0)**2
    ew2_cluster2 = np.linalg.norm(Wtilde2.T @ eigvecs2, axis=0)**2

    log_t_max = 6
    n_training_times = 1000
    log_training_times = np.linspace(0, log_t_max, n_training_times)
    training_times = 10**log_training_times

    loss_train = np.zeros(n_training_times)
    loss_train_cluster1 = np.zeros(n_training_times)
    loss_train_cluster2 = np.zeros(n_training_times)
    loss_train[0] = 1.0
    loss_train_cluster1[0] = 1.0
    loss_train_cluster2[0] = 1.0
    

    for tau_index in range(n_training_times-1):
        tau = training_times[tau_index+1]
        delta_loss = (1-np.exp(-4*Delta_t*lr*eigs*tau/d))/eigs*ew
        loss_train[tau_index+1] = 1 - np.sum(delta_loss)*Delta_t/d**2

        delta_loss_cluster1 = (1-np.exp(-4*Delta_t*lr*eigs1*tau/(weight*d)))*weight/eigs1*ew1_cluster1
        loss_train_cluster1[tau_index+1] = 1 - np.sum(delta_loss_cluster1)*Delta_t/d**2
        delta_loss_cluster2 = (1-np.exp(-4*Delta_t*lr*eigs2*tau/((1-weight)*d)))*(1-weight)/eigs2*ew2_cluster2
        loss_train_cluster2[tau_index+1] = 1 - np.sum(delta_loss_cluster2)*Delta_t/d**2

    return training_times, loss_train, loss_train_cluster1, loss_train_cluster2


def print_training_loss_curve(training_times, loss_train, loss_train_cluster1, loss_train_cluster2, fname):
    """Write training loss curve to file."""
    for i in range(len(training_times)):
        print(training_times[i], loss_train[i], loss_train_cluster1[i], loss_train_cluster2[i], file=fname)

# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    
    #output directory
    output_dir = f"results_spectrum/output_spectral_quantities_sampling_bias_t_diff={t_diffusion}_type={diffusion_type}/P={p}_D={d}_S={n_samples}/"
    os.makedirs(output_dir, exist_ok=True)

    # Output files for total and per-cluster spectra
    fname = open(os.path.join(output_dir, f"spectral_quantities_weight={weight1}_var1={var1}_var2={var2}_ortho={orthogonal_centroids}_batch.txt"), "a")
    fname_cluster1 = open(os.path.join(output_dir, f"spectral_quantities_cluster1_weight={weight1}_var1={var1}_var2={var2}_ortho={orthogonal_centroids}_batch.txt"), "a")
    fname_cluster2 = open(os.path.join(output_dir, f"spectral_quantities_cluster2_weight={weight1}_var1={var1}_var2={var2}_ortho={orthogonal_centroids}_batch.txt"), "a")
    
    print(f"\n{'='*80}")
    print(f"GEP Gaussian Mixture (BATCH PROCESSING + EMPIRICAL MEANS)")
    print(f"{'='*80}")
    print(f"Parameters: d={d}, p={p}, n_samples={n_samples}, n_realizations={n_realizations}")
    print(f"Batch size: {batch_size}, MC samples: {n_mc}, weight: {weight1}, variance ratio: {vr}")
    print(f"{'='*80}\n")
    
    for realization_idx in range(n_realizations):
        print(f"\n[Realization {realization_idx+1}/{n_realizations}]")
        
        # Generate samples from Gaussian mixture
        X = generate_samples_from_unbalanced_gaussian_mixture(
            d, n_samples, var1, var2, shift, sampling_bias=weight1 - 0.5,
            orthogonal_centroids=orthogonal_centroids
        )
        
        # Generate random weights
        W = generate_random_weights(p, d)
        
        # =====================================================================
        # PRECOMPUTE cluster-specific quantities (only 2 clusters!)
        # =====================================================================
        n_cluster1 = int(n_samples * weight1)
        
        # Cluster indices: 0 for cluster 1, 1 for cluster 2
        cluster_indices = np.array([0 if i < n_cluster1 else 1 for i in range(n_samples)])
        
        # Centroids for each cluster
        centroid1 = np.full(d, shift)
        if orthogonal_centroids:
            # Construct centroid2 orthogonal to centroid1
            # centroid1 = (shift1, shift1, ..., shift1)
            # For orthogonality: centroid1 · centroid2 = 0
            # Use: first half = +shift2, second half = -shift2 (so they cancel)
            if d % 2 == 0:
                # Even d: exactly half positive, half negative
                centroid2 = np.concatenate((-shift * np.ones(d // 2), shift * np.ones(d // 2)))
            else:
                # Odd d: (d-1)/2 positive, 1 zero, (d-1)/2 negative
                centroid2 = np.concatenate((-shift * np.ones(d // 2), np.array([0.0]), shift * np.ones(d // 2)))
        else:
            centroid2 = np.full(d, -shift)
        
        # Projected centroids: mu_lambda = W @ centroid / sqrt(d)
        mu_lambda_cluster1 = np.dot(W, centroid1) / np.sqrt(d)
        mu_lambda_cluster2 = np.dot(W, centroid2) / np.sqrt(d)
        
        # Gamma_t for each cluster
        Gamma_t1 = np.sqrt(var1 * adiff**2 + Delta_t)
        Gamma_t2 = np.sqrt(var2 * adiff**2 + Delta_t)
        
        # PRECOMPUTE Hermite integrals for both clusters (key optimization!)
        print(f"  Precomputing Hermite integrals for 2 clusters...")
        hermite_integrals_cluster1 = compute_hermite_integrals_GEP_hybrid(
            Gamma_t1, mu_lambda_cluster1, var1, adiff, p, n_mc, xi, gaussian_weights
        )
        hermite_integrals_cluster2 = compute_hermite_integrals_GEP_hybrid(
            Gamma_t2, mu_lambda_cluster2, var2, adiff, p, n_mc, xi, gaussian_weights
        )
        print(f"  ✓ Hermite integrals precomputed (saved {n_samples - 2} computations!)")
        
        # Create batches of sample indices
        n_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
        batches = [
            list(range(i * batch_size, min((i + 1) * batch_size, n_samples)))
            for i in range(n_batches)
        ]
        
        # Process batches in parallel
        print(f"  Processing {n_samples} samples in {n_batches} batches (batch_size={batch_size})...")
        checkpoint_interval = max(1, batch_size // 5)  # Print ~5 checkpoints per batch
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(
                batch_indices, X, W, cluster_indices,
                hermite_integrals_cluster1, hermite_integrals_cluster2,
                mu_lambda_cluster1, mu_lambda_cluster2,
                var1, var2, Gamma_t1, Gamma_t2,
                adiff, Delta_t, p, n_samples, d,
                batch_id=batch_idx+1, checkpoint_interval=checkpoint_interval
            ) for batch_idx, batch_indices in enumerate(batches)
        )
        
        # Unpack batch results: each element is (U_GEP, U_cluster1, U_cluster2)
        U_GEP_list = [r[0] for r in batch_results]
        U_cluster1_list = [r[1] for r in batch_results]
        U_cluster2_list = [r[2] for r in batch_results]
        
        # Sum all batch contributions
        U_GEP = np.sum(np.array(U_GEP_list), axis=0)
        U_cluster1 = np.sum(np.array(U_cluster1_list), axis=0)
        U_cluster2 = np.sum(np.array(U_cluster2_list), axis=0)
        print(f"  ✓ Computed U matrices ({p}×{p}): total, cluster1, cluster2")
        
        # Eigendecomposition for all three matrices
        print("  Computing eigenvalues...")
        eigs_GEP, eigvecs_GEP = np.linalg.eigh(U_GEP)
        eigs_cluster1, eigvecs_cluster1 = np.linalg.eigh(U_cluster1)
        eigs_cluster2, eigvecs_cluster2 = np.linalg.eigh(U_cluster2)
        print(f"  ✓ Total eigenvalues: [{eigs_GEP.min():.6f}, {eigs_GEP.max():.6f}]")
        print(f"  ✓ Cluster1 eigenvalues: [{eigs_cluster1.min():.6f}, {eigs_cluster1.max():.6f}]")
        print(f"  ✓ Cluster2 eigenvalues: [{eigs_cluster2.min():.6f}, {eigs_cluster2.max():.6f}]")
        
        # Save results for all three
        print_spectral_quantities(eigs_GEP, eigvecs_GEP, [mu_lambda_cluster1, mu_lambda_cluster2], fname, p)
        print_spectral_quantities(eigs_cluster1, eigvecs_cluster1, [mu_lambda_cluster1, mu_lambda_cluster2], fname_cluster1, p)
        print_spectral_quantities(eigs_cluster2, eigvecs_cluster2, [mu_lambda_cluster1, mu_lambda_cluster2], fname_cluster2, p)
        all_eigs_GEP[realization_idx] = eigs_GEP
        all_eigs_GEP_cluster1[realization_idx] = eigs_cluster1
        all_eigs_GEP_cluster2[realization_idx] = eigs_cluster2
    
    fname.close()
    fname_cluster1.close()
    fname_cluster2.close()

    # Compute and save training loss curves of last realization
    lr = 2e-4 * d/Delta_t
    print("\nComputing analytical training loss curves...")
    training_times, loss_train, loss_train_cluster1, loss_train_cluster2 = compute_analytical_curve_training_loss(
        eigs_GEP, eigvecs_GEP,
        eigs_cluster1, eigvecs_cluster1,
        eigs_cluster2, eigvecs_cluster2,
        d, p, Delta_t,
        [Gamma_t1, Gamma_t2],
        [hermite_integrals_cluster1[:, 2],
         hermite_integrals_cluster2[:, 2]],
        W, weight1, lr
    )
    fname_loss = open(os.path.join(output_dir, f"analytical_training_loss_curve_weight={weight1}_var1={var1}_var2={var2}_ortho={orthogonal_centroids}_batch.txt"), "a")
    print_training_loss_curve(training_times, loss_train, loss_train_cluster1, loss_train_cluster2, fname_loss)
    fname_loss.close()
    print("✓ Analytical training loss curves saved.")

    ## savefig of training loss curves
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(training_times), loss_train, label="Total", color="black")
    plt.plot(np.log10(training_times), loss_train_cluster1, label="Cluster 1", color="blue", linestyle="--")
    plt.plot(np.log10(training_times), loss_train_cluster2, label="Cluster 2", color="green", linestyle=":")
    plt.xlabel("log Training Time", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title(f"Analytical Training Loss Curve - Gaussian Mixture (Batch + Empirical)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analytical_training_loss_curve_N={d}_P={p}_S={n_samples}_batch.png"), dpi=150)
    #plt.show()
    
    # Final summary
    print(f"\n{'='*80}")
    print("All realizations complete!")
    print(f"{'='*80}")
    
    eigss_GEP = np.sort(all_eigs_GEP.flatten())
    eigss_GEP_cluster1 = np.sort(all_eigs_GEP_cluster1.flatten())
    eigss_GEP_cluster2 = np.sort(all_eigs_GEP_cluster2.flatten())
    print(f"Total eigenvalues: {len(eigss_GEP)}")
    print(f"Range: [{eigss_GEP.min():.6f}, {eigss_GEP.max():.6f}]")

    print(f"\n Total execution time (in seconds) : {time.time() - start_time:.2f} s")
    
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        np.log10(eigss_GEP[eigss_GEP > 0][:-2*n_realizations]),
        bins=5*int(np.sqrt(p)+0.5),
        density=True,
        alpha=0.8,
        edgecolor='black'
    )
    plt.hist(
        np.log10(eigss_GEP_cluster1[eigss_GEP_cluster1 > 0][:-2*n_realizations]),
        bins=5*int(np.sqrt(p)+0.5),
        density=True,
        alpha=0.3,
        edgecolor='blue'
    )
    plt.hist(
        np.log10(eigss_GEP_cluster2[eigss_GEP_cluster2 > 0][:-2*n_realizations]),
        bins=5*int(np.sqrt(p)+0.5),
        density=True,
        alpha=0.15,
        edgecolor='green'
    )
    plt.xlabel("log₁₀(Eigenvalue)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Eigenvalue Distribution - Gaussian Mixture (Batch + Empirical)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ## output directory figures
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"eigenvalue_dist_mixture_N={d}_P={p}_S={n_samples}_batch.png"), dpi=150)
    plt.show()
    
