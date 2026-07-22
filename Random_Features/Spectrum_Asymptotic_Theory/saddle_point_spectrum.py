#!/usr/bin/env python3
"""
================================================================================
 Asymptotic (self-consistent) spectrum of the diffusion feature matrix U
================================================================================

OUTLINE OF THE SCRIPT
---------------------
This script computes the **asymptotic spectral density** of the random-feature
covariance matrix `U` of a two-cluster Gaussian mixture — a minimal model of
*class imbalance* in diffusion generative models — by solving the
**self-consistent (saddle-point) equations** of random-matrix theory, instead of
diagonalizing sampled matrices. From the spectrum it also derives the
**fully-analytical training-loss curve**.

The mixture centroids can be in one of two regimes, selected automatically from
`centroids_factor` and the dimension `d`:

  - "extensive"  : centroid norm is O(\sqrt{d}). The Hermite coefficients become functions of the projected centroids \mu and the saddle-point equations involve 2x2 matrices
                   (`g_gamma`, `g_omega`). Solved by `solve_equations_extensive_centroids`.
  - "finite"     : centroid norm is O(1). The Hermite coefficients are *scalars*
                   and the saddle-point equations reduce to two scalar unknowns
                   (`g_Psi`, `g_Omega`). Solved by `solve_equations_finite_centroids`.

For the selected regime the script:

  1. Computes the Hermite coefficients (alpha, beta_tilde, gamma, beta) that
     describe how `tanh` acts on Gaussian inputs, by trapezoidal integration.
  2. Solves the self-consistent equations for the resolvent `g(z)` along a grid
     of eigenvalues `lambda`, tracking the solution from large imaginary part
     down to the real axis (homotopy continuation in the small imaginary part).
     The spectral density is `rho(lambda) = Im g(lambda) / pi`.
  3. (finite regime) Locates the spectral edges `lambda_star`.
  4. Integrates the resulting density into the analytical training-loss curve.
  5. Writes the density, the edges, and the loss curve to text files under
     `results_asymptotic/`.

--------------------------------------------------------------------------------
HOW TO RUN THE SCRIPT
--------------------------------------------------------------------------------
Basic run with defaults (finite-centroid regime):

    python saddle_point_spectrum.py

Custom run (extensive regime is triggered by a larger centroid factor):

    python saddle_point_spectrum.py -d 100 -chi_p 60 -chi_s 30 -centroids_factor 1.0 -t 0.01

See all options:

    python saddle_point_spectrum.py --help

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    numpy, scipy   (matplotlib only needed if you re-enable the optional
    plotting blocks, which are commented out).

--------------------------------------------------------------------------------
STRUCTURE OF THE OUTPUT FOLDER
--------------------------------------------------------------------------------
A nested folder is created under `results_asymptotic/` whose name encodes the
regime and the main parameters, e.g.:

    results_asymptotic/solver_saddle_point_equations_finite/P=6000_D=100_S=3000/centroid=0.1_t=0.01/
        rho_lambda_analytical_weight=..._ortho.txt          (eigenvalue, density, ...)
        lambda_star_analytical_weight=..._ortho.txt         (spectral edges; finite only)
        training_loss_fully_analytical_weight=..._ortho.txt (training time, loss)
================================================================================
"""

import os
import time
import secrets
import argparse

import numpy as np
import matplotlib.pyplot as plt   # only used by the optional plotting blocks (commented out)
from scipy.optimize import root


# =============================================================================
# 1. COMMAND-LINE ARGUMENTS
# -----------------------------------------------------------------------------
# Every knob of the experiment is exposed here. Run with --help to see them all.
# =============================================================================
parser = argparse.ArgumentParser(
    description='Solve the self-consistent saddle-point equations for the spectrum of U.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-d', type=int, default=100, help='Dimensionality of the data.')
parser.add_argument('-chi_p', type=float, default=60.0, help='Value of chi_p parameter.')
parser.add_argument('-chi_s', type=float, default=30.0, help='Value of chi_s parameter.')
parser.add_argument('-weight', type=float, default=0.5, help='Weight for the mixture of clusters.')
parser.add_argument('-centroids_factor', type=float, default=0.1, help='Factor to scale the centroids of the clusters.')
parser.add_argument('-var0', type=float, default=0.5, help='Variance of the first cluster.')
parser.add_argument('-var1', type=float, default=0.25, help='Variance of the second cluster.')
parser.add_argument('-t', type=float, default=0.01, help='Time parameter for the diffusion process.')
parser.add_argument('-npoints_integration', type=int, default=5000, help='Number of integration points.')

args = parser.parse_args()


# =============================================================================
# 2. DERIVED PARAMETERS AND CONFIGURATION
# -----------------------------------------------------------------------------
# Diffusion coefficients, the numerical-integration grid, the automatic choice
# of centroid regime, and the solver / training-loss settings.
# =============================================================================

## input parameters

d = args.d
chi_p = args.chi_p
chi_s = args.chi_s
weight = args.weight
centroids_factor = args.centroids_factor
vars = np.array([args.var0, args.var1])
t = args.t

## diffusion parameters

at = np.exp(-t)
delta_t = 1-at**2
Gammas = np.sqrt(vars*at**2+delta_t)

## parameters for numerical integration

npoints_integration = args.npoints_integration
lower_bound_integration = -10.0
upper_bound_integration = 10.0
xi = np.linspace(lower_bound_integration, upper_bound_integration, npoints_integration)
mu_interval = xi*np.sqrt(centroids_factor)

## activation function

phi = lambda x: np.tanh(x)

## rng

seed = secrets.randbits(128)
rng = np.random.default_rng(seed=seed)

## mode: finite or extensive centroid norm

if centroids_factor > np.sqrt(10 / d):
    # extensive centroid norm (here sorted out as when centroid norm is at least one order of magnitude larger than unity)
    mode_centroid = "extensive"
    cf = centroids_factor
else:
    # finite centroid norm
    mode_centroid = "finite"
    cf = 0.0

## parameters for solver


if mode_centroid == "extensive":
    n_values = 140
    n_epsilon = 50
    lambda_values = np.logspace(-5, 1.7, n_values)
    epsilon_protocol = np.concatenate((np.logspace(2, -9, n_epsilon), [1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 0]))
else:
    n_iter = 100
    n_values = 10000
    lambda_values = np.logspace(-8, 1.7, n_values)

## parameters for estimation of fully analytical training loss

log_t_max = 10
n_training_times = 1100
log_training_times = np.linspace(0, log_t_max, n_training_times)
training_times = np.concatenate(([0], 10**log_training_times))
lr = 5e-5
threshold = 1e-6

# =============================================================================
# 3. HERMITE COEFFICIENTS
# -----------------------------------------------------------------------------
# Gaussian expectations of tanh(...) that characterize the non-linearity. In the
# "extensive" regime each returns a mu-dependent VECTOR (indexed by the projected
# centroid grid mu_interval); in the "finite" regime each returns a SCALAR.
# =============================================================================

def alpha_trapz(mode, Gammas, at, cluster_index):

    std = Gammas[cluster_index]

    inv_2pi_sqrt = 1/np.sqrt(2*np.pi)

    if mode == "extensive":

        ## in this case it computes the vector of hermite coefficients

        gaussian_weigth = np.outer(np.ones(mu_interval.size), np.exp(-0.5*xi**2))

        return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*phi(std*xi+mu_interval[:, np.newaxis]*at), xi, axis=1)
    
    else:

        ## in this case it computes the scalar hermite coefficient

        gaussian_weigth = np.exp(-0.5*xi**2)

        return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*phi(std*xi), xi)

def beta_tilde_trapz(mode, Gammas, at, cluster_index):

    std = Gammas[cluster_index]

    inv_2pi_sqrt = 1/np.sqrt(2*np.pi)

    if mode == "extensive":

        gaussian_weigth = np.outer(np.ones(mu_interval.size), np.exp(-0.5*xi**2))

        return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*phi(std*xi+mu_interval[:, np.newaxis]*at)**2, xi, axis=1)
    
    else:

        gaussian_weigth = np.exp(-0.5*xi**2)

        return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*phi(std*xi)**2, xi)
    
def gamma_trapz(mode, Gammas, at, vars, cluster_index):

    std = Gammas[cluster_index]

    inv_2pi_sqrt = 1/np.sqrt(2*np.pi)

    if mode == "extensive":

        gaussian_weigth = np.outer(np.ones(mu_interval.size), np.exp(-0.5*xi**2))

        return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*np.outer(np.ones(mu_interval.size), xi)*phi(std*xi+mu_interval[:, np.newaxis]*at), xi, axis=1)*np.sqrt(vars[cluster_index])*at/Gammas[cluster_index]
    
    else:

        gaussian_weigth = np.exp(-0.5*xi**2)

        return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*xi*phi(std*xi), xi)*np.sqrt(vars[cluster_index])*at/Gammas[cluster_index]
    
def beta_trapz_2d(at, delta_t, mode, var, tol=1e-5, n_init=1201, n_max=9601, z_bound=10.0):
    """
    Deterministic computation of beta using nested trapezoidal integration.
    
    beta = E_{z1}[ (E_{z2}[tanh(s1*z1 + s2*z2 + shift)])^2 ],
    with s1 = sqrt(var)*at, s2 = sqrt(delta_t), shift = at*mu.

    The grid is refined until two consecutive estimates differ by < tol.
    """

    if n_init < 3:
        n_init = 3
    if n_init % 2 == 0:
        n_init += 1

    sqrt_2pi_inv = 1.0 / np.sqrt(2.0 * np.pi)
    s1 = np.sqrt(var) * at
    s2 = np.sqrt(delta_t)

    if mode == "extensive":
        mu = np.linspace(-lower_bound_integration*np.sqrt(centroids_factor), upper_bound_integration*np.sqrt(centroids_factor), npoints_integration)
        shift = at * mu
    else:
        shift = 0.0

    def _beta_on_grid(n_points):
        z = np.linspace(-z_bound, z_bound, n_points)
        pdf = sqrt_2pi_inv * np.exp(-0.5 * z**2)

        # Inner expectation over z2 for each z1 (vectorized on the 2D mesh).
        arg = s1 * z[:, np.newaxis] + s2 * z[np.newaxis, :] + shift
        inner = np.trapezoid(pdf[np.newaxis, :] * np.tanh(arg), z, axis=1)

        # Outer expectation over z1.
        return np.trapezoid(pdf * inner**2, z)

    n_points = n_init
    beta_prev = _beta_on_grid(n_points)
    err_est = np.inf

    while n_points < n_max:
        # Refine by doubling intervals while keeping odd number of points.
        n_points = 2 * (n_points - 1) + 1
        beta_curr = _beta_on_grid(n_points)
        err_est = np.abs(beta_curr - beta_prev)
        if err_est < tol:
            return beta_curr, n_points, err_est
        beta_prev = beta_curr

    return beta_prev, n_points, err_est

def beta_multiple_values_fixed_grid(mu_values, var, n_points, z_bound=10.0):
    """Compute beta(mu) on a list of mu values with deterministic 2D trapezoid integration."""
    z = np.linspace(-z_bound, z_bound, n_points)
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    scale_1 = np.sqrt(var) * at
    scale_2 = np.sqrt(delta_t)

    betas = np.empty(mu_values.size)
    for i, mu_i in enumerate(mu_values):
        shift = at * mu_i
        arg = scale_1 * z[:, np.newaxis] + scale_2 * z[np.newaxis, :] + shift
        inner = np.trapezoid(pdf[np.newaxis, :] * np.tanh(arg), z, axis=1)
        betas[i] = np.trapezoid(pdf * inner**2, z)

    return betas

def beta_multiple_values(mu_interval, var, n_points_for_mu_grid=301, n_points_2d=801, z_bound=10.0):
    """
    Build beta(mu) deterministically on a sparse mu-grid and interpolate
    to the full mu_interval to keep runtime manageable.
    """
    mu_nodes = np.linspace(mu_interval[0], mu_interval[-1], n_points_for_mu_grid)
    beta_nodes = beta_multiple_values_fixed_grid(mu_nodes, var, n_points=n_points_2d, z_bound=z_bound)
    if n_points_for_mu_grid < mu_interval.size:
        betas_interp = np.interp(mu_interval, mu_nodes, beta_nodes)
    else:
        betas_interp = beta_nodes
    return betas_interp, mu_nodes, beta_nodes

def D_call(weight, at, vars, beta_tildes, betas, gammas):

    var0 = vars[0]
    var1 = vars[1]

    beta_tilde0 = beta_tildes[0]
    beta_tilde1 = beta_tildes[1]

    beta0 = betas[0]
    beta1 = betas[1]

    gamma0 = gammas[0]
    gamma1 = gammas[1]

    return weight*(beta_tilde0-beta0-delta_t*gamma0**2/(var0*at**2)) + (1-weight)*(beta_tilde1-beta1-delta_t*gamma1**2/(var1*at**2))

# =============================================================================
# 4. EXTENSIVE-CENTROID SADDLE-POINT EQUATIONS
# -----------------------------------------------------------------------------
# Regime where the centroid norm grows with d. The resolvent involves the 2x2
# matrices g_gamma / g_omega and the auxiliary functions `theta` and `ell`.
# `solve_equations_extensive_centroids` sweeps lambda and returns the resolvent
# and Omega-density arrays.
# =============================================================================

## Theta (it is only used in the extensive centroid case)

def theta(z, num, alphas, betas, gammas, betas_tildes, g_bubbas, g_gamma_matrix, g_omega_matrix):

    alpha0 = alphas[0]
    alpha1 = alphas[1]

    beta0 = betas[0]
    beta1 = betas[1]

    beta_tilde0 = betas_tildes[0]
    beta_tilde1 = betas_tildes[1]

    gamma0 = gammas[0]
    gamma1 = gammas[1]

    g_bubba0 = g_bubbas[0]
    g_bubba1 = g_bubbas[1]

    g_omega0 = g_omega_matrix[0][0]
    g_omega1 = g_omega_matrix[1][1]

    num0 = beta0[:, np.newaxis]-alpha0[:, np.newaxis]**2-gamma0[:, np.newaxis]**2
    #num0 = beta0[:, np.newaxis]-gamma0[:, np.newaxis]**2
    denom0 = 1 + chi_p / chi_s * g_bubba0 + chi_p / chi_s * g_omega0

    num1 = beta1[np.newaxis, :]-alpha1[np.newaxis, :]**2-gamma1[np.newaxis, :]**2
    #num1 = beta1[np.newaxis, :]-gamma1[np.newaxis, :]**2
    denom1 = 1 + chi_p / chi_s * g_bubba1 + chi_p / chi_s * g_omega1

    R_easy_val = weight*num0/denom0 + (1-weight)*num1/denom1

    g_gamma_matrix_inverse = np.linalg.inv(g_gamma_matrix)

    y_matrix = g_gamma_matrix_inverse - g_gamma_matrix_inverse @ g_omega_matrix @ g_gamma_matrix_inverse 

    R_hard_val = np.dot(np.array(gammas).T, np.dot(y_matrix, np.array(gammas)))/chi_p

    
    var0 = vars[0]
    var1 = vars[1]

    D_call_val = weight*(beta_tilde0[:, np.newaxis]-beta0[:, np.newaxis]-delta_t*gamma0[:, np.newaxis]**2/(var0*at**2)) + (1-weight)*(beta_tilde1[np.newaxis, :]-beta1[np.newaxis, :]-delta_t*gamma1[np.newaxis, :]**2/(var1*at**2))
   
    denom_of_the_integral = D_call_val - z + R_easy_val + R_hard_val

    gaussian_weight = np.exp(-0.5 * centroids_factor**(-2) * mu_interval**2) / np.sqrt(2.0 * np.pi * centroids_factor**2)

    integrand = num/denom_of_the_integral

    inner_integral = np.trapezoid(integrand*gaussian_weight[np.newaxis, :], mu_interval, axis=1)
    
    outer_integral = np.trapezoid(inner_integral*gaussian_weight, mu_interval)

    return outer_integral

## ell function (it is only used in the extensive centroid case)

def ell(g_bubbas, g_gamma_matrix, g_omega_matrix_diag, cluster_index):

    w = weight if cluster_index == 0 else (1-weight)

    g_gamma_matrix_inverse = np.linalg.inv(g_gamma_matrix)

    term1 = w * delta_t / (vars[cluster_index]*at**2)

    term2 = g_gamma_matrix_inverse[cluster_index, cluster_index]/chi_p

    term3 = w / (1 + chi_p / chi_s * g_bubbas[cluster_index] + chi_p / chi_s * g_omega_matrix_diag[cluster_index])

    return (term1 + term2 + term3)*chi_p

def equations_extensive_centroids(variables, z, params):

    g_bubba0real, g_bubba0imag, g_bubba1real, g_bubba1imag, g_gamma00real, g_gamma01real, g_gamma11real, g_gamma00imag, g_gamma01imag, g_gamma11imag, g_omega00real, g_omega01real, g_omega11real, g_omega00imag, g_omega01imag, g_omega11imag = variables

    g_bubba0 = complex(g_bubba0real, g_bubba0imag)
    g_bubba1 = complex(g_bubba1real, g_bubba1imag)
    g_gamma00 = complex(g_gamma00real, g_gamma00imag)
    g_gamma01 = complex(g_gamma01real, g_gamma01imag)
    g_gamma11 = complex(g_gamma11real, g_gamma11imag)
    g_omega00 = complex(g_omega00real, g_omega00imag)
    g_omega01 = complex(g_omega01real, g_omega01imag)
    g_omega11 = complex(g_omega11real, g_omega11imag)

    g_bubbas = [g_bubba0, g_bubba1]
    g_gamma_matrix = np.array([[g_gamma00, g_gamma01], [g_gamma01, g_gamma11]])
    ## ensure invertibility of gamma matrix by adding a small regularization term if needed
    if np.linalg.norm(g_gamma_matrix[0, 0]-g_gamma_matrix[0, 1]) < 1e-16 or np.linalg.norm(g_gamma_matrix[1, 1]-g_gamma_matrix[0, 1]) < 1e-16:
        g_gamma_matrix += 1e-16*np.eye(2)
    g_gamma_matrix_inverse = np.linalg.inv(g_gamma_matrix)

    g_omega_matrix = np.array([[g_omega00, g_omega01], [g_omega01, g_omega11]])
    ## ensure invertibility of omega matrix by adding a small regularization term if needed
    if np.linalg.norm(g_omega_matrix[0, 0]-g_omega_matrix[0, 1]) < 1e-16 or np.linalg.norm(g_omega_matrix[1, 1]-g_omega_matrix[0, 1]) < 1e-16:
        g_omega_matrix += 1e-16*np.eye(2)

    alpha0 = params["alpha0"]
    alpha1 = params["alpha1"]

    alphas = [alpha0, alpha1]

    beta0 = params["beta0"]
    beta1 = params["beta1"]

    betas = [beta0, beta1]

    gamma0 = params["gamma0"]
    gamma1 = params["gamma1"]

    gammas = [gamma0, gamma1]

    beta_tilde0 = params["beta_tilde0"]
    beta_tilde1 = params["beta_tilde1"]

    betas_tildes = [beta_tilde0, beta_tilde1]

    equation_g_bubba0 = g_bubba0 - theta(z, beta0[:, np.newaxis]-alpha0[:, np.newaxis]**2-gamma0[:, np.newaxis]**2, alphas, betas, gammas, betas_tildes, g_bubbas, g_gamma_matrix, g_omega_matrix)

    equation_g_bubba1 = g_bubba1 - theta(z, beta1[np.newaxis, :]-alpha1[np.newaxis, :]**2-gamma1[np.newaxis, :]**2, alphas, betas, gammas, betas_tildes, g_bubbas, g_gamma_matrix, g_omega_matrix)

    equation_g_gamma00 = g_gamma_matrix[0, 0] - theta(z, gamma0[:, np.newaxis]**2, alphas, betas, gammas, betas_tildes, g_bubbas, g_gamma_matrix, g_omega_matrix)

    equation_g_gamma11 = g_gamma_matrix[1, 1] - theta(z, gamma1[np.newaxis, :]**2, alphas, betas, gammas, betas_tildes, g_bubbas, g_gamma_matrix, g_omega_matrix)

    equation_g_gamma01 = g_gamma_matrix[0, 1] - theta(z, gamma0[:, np.newaxis]*gamma1[np.newaxis, :], alphas, betas, gammas, betas_tildes, g_bubbas, g_gamma_matrix, g_omega_matrix)

    equation_g_omega00 = g_omega_matrix[0, 0] - ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 1)/(ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 0)*ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 1)-g_gamma_matrix_inverse[0, 1]**2)

    equation_g_omega11 = g_omega_matrix[1, 1] - ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 0)/(ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 0)*ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 1)-g_gamma_matrix_inverse[0, 1]**2)

    equation_g_omega01 = g_omega_matrix[0, 1] + g_gamma_matrix_inverse[0, 1]/(ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 0)*ell(g_bubbas, g_gamma_matrix, [g_omega00, g_omega11], 1)-g_gamma_matrix_inverse[0, 1]**2)
    

    return np.array([equation_g_bubba0.real, equation_g_bubba0.imag, equation_g_bubba1.real, equation_g_bubba1.imag, equation_g_gamma00.real, equation_g_gamma00.imag, equation_g_gamma01.real, equation_g_gamma01.imag, equation_g_gamma11.real, equation_g_gamma11.imag, equation_g_omega00.real, equation_g_omega00.imag, equation_g_omega01.real, equation_g_omega01.imag, equation_g_omega11.real, equation_g_omega11.imag])

def compute_g_extensive_centroids(solution, z, params):

    g_bubba0_real = solution.x[0]
    g_bubba0_imag = solution.x[1]
    g_bubba1_real = solution.x[2]
    g_bubba1_imag = solution.x[3]
    g_gamma00_real = solution.x[4]
    g_gamma01_real = solution.x[5]
    g_gamma11_real = solution.x[6]
    g_gamma00_imag = solution.x[7]
    g_gamma01_imag = solution.x[8]
    g_gamma11_imag = solution.x[9]
    g_omega00_real = solution.x[10]
    g_omega00_imag = solution.x[11]
    g_omega01_real = solution.x[12]
    g_omega01_imag = solution.x[13]
    g_omega11_real = solution.x[14]
    g_omega11_imag = solution.x[15]

    g_bubba0 = complex(g_bubba0_real, g_bubba0_imag)
    g_bubba1 = complex(g_bubba1_real, g_bubba1_imag)
    g_gamma00 = complex(g_gamma00_real, g_gamma00_imag)
    g_gamma11 = complex(g_gamma11_real, g_gamma11_imag)
    g_gamma01 = complex(g_gamma01_real, g_gamma01_imag)
    g_omega00 = complex(g_omega00_real, g_omega00_imag)
    g_omega01 = complex(g_omega01_real, g_omega01_imag)
    g_omega11 = complex(g_omega11_real, g_omega11_imag)

    alphas = [params["alpha0"], params["alpha1"]]
    betas = [params["beta0"], params["beta1"]]
    gammas = [params["gamma0"], params["gamma1"]]
    betas_tildes = [params["beta_tilde0"], params["beta_tilde1"]]

    g=theta(z, 1, alphas, betas, gammas, betas_tildes, [g_bubba0, g_bubba1], [[g_gamma00, g_gamma01], [g_gamma01, g_gamma11]], np.array([[g_omega00, g_omega01], [g_omega01, g_omega11]]))

    return g

def solve_equations_extensive_centroids(params):

    alpha0 = params["alpha0"]
    alpha1 = params["alpha1"]

    beta0 = params["beta0"]
    beta1 = params["beta1"]

    gamma0 = params["gamma0"]
    gamma1 = params["gamma1"]

    g_real = np.zeros(n_values)
    g_imag = np.zeros(n_values)
    g_Omega00_real = np.zeros(n_values)
    g_Omega00_imag = np.zeros(n_values)
    g_Omega01_real = np.zeros(n_values)
    g_Omega01_imag = np.zeros(n_values)
    g_Omega11_real = np.zeros(n_values)
    g_Omega11_imag = np.zeros(n_values)

    gaussian_weight = np.exp(-0.5 * centroids_factor**(-2) * mu_interval**2) / np.sqrt(2.0 * np.pi * centroids_factor**2)

    for count, lambda_value in enumerate(lambda_values):
        
        z = lambda_value + epsilon_protocol[0]*1j

        ## initialization at the beginning of the protocol

        bubba_factor0 = np.trapezoid((beta0-gamma0**2-alpha0**2)*gaussian_weight, mu_interval)
        bubba_factor1 = np.trapezoid((beta1-gamma1**2-alpha1**2)*gaussian_weight, mu_interval)

        g_gamma_matrix_00_factor = np.trapezoid((gamma0**2)*gaussian_weight, mu_interval)
        g_gamma_matrix_11_factor = np.trapezoid((gamma1**2)*gaussian_weight, mu_interval)
        g_gamma_matrix_01_factor = np.trapezoid(gamma0*gaussian_weight, mu_interval)*np.trapezoid(gamma1*gaussian_weight, mu_interval)

        g_bubbas_initial = -np.array([bubba_factor0/z, bubba_factor1/z])
        g_gamma_matrix_initial = -np.array([[g_gamma_matrix_00_factor/z, g_gamma_matrix_01_factor/z], [g_gamma_matrix_01_factor/z, g_gamma_matrix_11_factor/z]])
        g_gamma_matrix_initial_inverse = np.linalg.inv(g_gamma_matrix_initial)

        determinant = ell(g_bubbas_initial, g_gamma_matrix_initial, [g_gamma_matrix_initial[0, 0], g_gamma_matrix_initial[1, 1]], 0)*ell(g_bubbas_initial, g_gamma_matrix_initial, [g_gamma_matrix_initial[0, 0], g_gamma_matrix_initial[1, 1]], 1)-g_gamma_matrix_initial_inverse[0, 1]**2

        if determinant == 0:

            determinant += 1e-16
        
        g_omega_matrix_initial_00 = ell(g_bubbas_initial, g_gamma_matrix_initial, [g_gamma_matrix_initial[0, 0], g_gamma_matrix_initial[1, 1]], 1)/determinant

        g_omega_matrix_initial_11 = ell(g_bubbas_initial, g_gamma_matrix_initial, [g_gamma_matrix_initial[0, 0], g_gamma_matrix_initial[1, 1]], 0)/determinant

        g_omega_matrix_initial_01 = -g_gamma_matrix_initial_inverse[0, 1]/determinant

        g_omega_matrix_initial = np.array([[g_omega_matrix_initial_00, g_omega_matrix_initial_01], [g_omega_matrix_initial_01, g_omega_matrix_initial_11]])

        input_variables = np.concatenate((
                                        [g_bubbas_initial[0].real, g_bubbas_initial[1].real],
                                        [g_bubbas_initial[0].imag, g_bubbas_initial[1].imag],
                                        [g_gamma_matrix_initial[0, 0].real, g_gamma_matrix_initial[0, 1].real, g_gamma_matrix_initial[1, 1].real],
                                        [g_gamma_matrix_initial[0, 0].imag, g_gamma_matrix_initial[0, 1].imag, g_gamma_matrix_initial[1, 1].imag], [g_omega_matrix_initial[0, 0].real, g_omega_matrix_initial[0, 1].real, g_omega_matrix_initial[1, 1].real], [g_omega_matrix_initial[0, 0].imag, g_omega_matrix_initial[0, 1].imag, g_omega_matrix_initial[1, 1].imag]
                                        ))
  

        ## solve equations at first point of the protocol

        solution = root(equations_extensive_centroids, input_variables, args=(z, params), method='lm')

        ## cycle over the protocol

        g = compute_g_extensive_centroids(solution, z, params)

        input_variables = solution.x

        flag = 0

        for epsilon in epsilon_protocol[1:]:

            z = lambda_value + epsilon*1j

            g_previous = np.copy(g)
            solution_previous = np.copy(input_variables)

            solution = root(equations_extensive_centroids, input_variables, args=(z, params), method='lm')

            input_variables = solution.x

            g = compute_g_extensive_centroids(solution, z, params)

            ## if imaginary part of g gets negative, exit the cycle
            if g.imag < 0:

                flag = 1

                g_real[count] = g_previous.real
                g_imag[count] = g_previous.imag
                g_Omega00_real[count] = solution_previous[10]
                g_Omega00_imag[count] = solution_previous[11]
                g_Omega01_real[count] = solution_previous[12]
                g_Omega01_imag[count] = solution_previous[13]
                g_Omega11_real[count] = solution_previous[14]
                g_Omega11_imag[count] = solution_previous[15]

                break

        if flag != 1:

            g_real[count] = g.real
            g_imag[count] = g.imag
            g_Omega00_real[count] = solution.x[10]
            g_Omega00_imag[count] = solution.x[11]
            g_Omega01_real[count] = solution.x[12]
            g_Omega01_imag[count] = solution.x[13]
            g_Omega11_real[count] = solution.x[14]
            g_Omega11_imag[count] = solution.x[15]

        ## print every 10 lambda values the progress of the protocol

        if count % 10 == 0:

            print(f"Finished lambda value {count+1}/{n_values} with lambda={lambda_value:.5f} and g={complex(g_real[count], g_imag[count]):.3e}, epsilon={epsilon:.2e}")

    return g_real, g_imag, g_Omega00_real, g_Omega00_imag, g_Omega01_real, g_Omega01_imag, g_Omega11_real, g_Omega11_imag

# =============================================================================
# 5. FINITE-CENTROID SADDLE-POINT EQUATIONS
# -----------------------------------------------------------------------------
# Regime where the centroid norm is O(1). The self-consistent equations reduce
# to two scalar unknowns (g_Psi, g_Omega). `solve_equations_finite_centroids`
# sweeps lambda and returns the resolvent and Omega-density arrays; the
# `*_for_spectral_edges_*` / `spectral_edge_*` helpers locate the band edges.
# =============================================================================

def equations_finite_centroids(variables, z_real, z_imag, params):

    g_Psi_real, g_Psi_img, g_Omega_real, g_Omega_img = variables

    g_Psi = complex(g_Psi_real, g_Psi_img)
    g_Omega = complex(g_Omega_real, g_Omega_img)

    weight = params["weight"]

    at = params["at"]
    delta_t = params["delta_t"]

    var0 = params["var0"]
    var1 = params["var1"]

    beta0_scalar = params["beta0"]
    beta1_scalar = params["beta1"]

    gamma0_scalar = params["gamma0"]
    gamma1_scalar = params["gamma1"]

    beta_tilde0_scalar = params["beta_tilde0"]
    beta_tilde1_scalar = params["beta_tilde1"]

    D_call_scalar = D_call(weight, at, vars, [beta_tilde0_scalar, beta_tilde1_scalar], [beta0_scalar, beta1_scalar], [gamma0_scalar, gamma1_scalar])

    def R_easy_scalar_for_Psi(g_Psi, g_Omega):

        num0 = beta0_scalar-gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = beta1_scalar-gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-gamma1_scalar**2) + (chi_p / chi_s) * gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1
    
    def R_easy_scalar_for_Omega(g_Psi, g_Omega):

        num0 = gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-gamma1_scalar**2) + (chi_p / chi_s) *gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1

    eqs = np.empty(4)

    z = complex(z_real, z_imag)

    eq_Psi_complex = (1-1.0/chi_p)/g_Psi + (1.0/chi_p)*g_Omega/(g_Psi**2) - (D_call_scalar - z) - R_easy_scalar_for_Psi(g_Psi, g_Omega)

    eq_Omega_complex = 1/g_Omega/chi_p - delta_t * (gamma0_scalar**2*weight/(at**2*var0) + gamma1_scalar**2*(1-weight)/(at**2*var1)) - (1.0/chi_p)*(1.0/g_Psi)-R_easy_scalar_for_Omega(g_Psi, g_Omega)
    
    eqs[0] = eq_Omega_complex.real
    eqs[1] = eq_Omega_complex.imag
    eqs[2] = eq_Psi_complex.real
    eqs[3] = eq_Psi_complex.imag


    return eqs

def solve_equations_finite_centroids(params):
    """Sweep the eigenvalue grid and solve the finite-centroid equations.

    This is the finite-regime counterpart of `solve_equations_extensive_centroids`.
    For each lambda it follows the resolvent from a large imaginary part of z down
    to the real axis (homotopy continuation: the previous solution seeds the next
    `root` call), so the branch is tracked continuously.

    Returns four arrays of length `n_values`:
        real_g, rho, real_g_Omega, rho_Omega
    where rho = |Im g| / pi and rho_Omega = |Im g_Omega| / pi.
    """
    real_g = np.zeros(n_values)
    rho = np.zeros(n_values)
    real_g_Omega = np.zeros(n_values)
    rho_Omega = np.zeros(n_values)

    for count, z_real in enumerate(lambda_values[1:]):

        z = z_real+100j

        g_Psi_initial = 1/z
        g_Omega_initial = 1/z

        guess = np.array([g_Psi_initial.real, g_Psi_initial.imag, g_Omega_initial.real, g_Omega_initial.imag])

        # Coarse continuation from large imaginary part (im = n_iter, ..., 1).
        for im in range(n_iter, 0, -1):

            solution = root(equations_finite_centroids, guess, args=(z_real,-im, params), method='lm')

            guess = solution.x

        # Fine approach to the real axis.
        for im in [0.9,0.7,0.5,0.3,0.1,0.01,0.001,0.0001,0.00001,1e-6,1e-7,1e-8,1e-9, 0.0]:

            solution = root(equations_finite_centroids, guess, args=(z_real,-im, params), method='lm')

            guess = solution.x

        # Final solve exactly on the real axis.
        solution = root(equations_finite_centroids, guess, args=(z_real, 0.0, params), method='lm')
        guess = solution.x

        real_g[count] = guess[0]
        rho[count] = np.abs(guess[1])/np.pi
        real_g_Omega[count] = guess[2]
        rho_Omega[count] = np.abs(guess[3])/np.pi

        ## print progress every 500 lambda values
        if count % 500 == 0:
            print(f"Finished lambda value {count+1}/{n_values} with lambda={z_real:.5f} and rho={rho[count]:.3e}")

    return real_g, rho, real_g_Omega, rho_Omega

def equations_for_spectral_edges_finite_centroids(variables, params):

    g_Psi_star, g_Omega_star = variables

    weight = params["weight"]

    at = params["at"]
    delta_t = params["delta_t"]

    var0 = params["var0"]
    var1 = params["var1"]

    beta0_scalar = params["beta0"]
    beta1_scalar = params["beta1"]

    gamma0_scalar = params["gamma0"]
    gamma1_scalar = params["gamma1"]

    beta_tilde0_scalar = params["beta_tilde0"]
    beta_tilde1_scalar = params["beta_tilde1"]

    #diag = D_call(weight, at, vars, [beta_tilde0_scalar, beta_tilde1_scalar], [beta0_scalar, beta1_scalar], [gamma0_scalar, gamma1_scalar])

    def omega_tilde_star(g_Psi, g_Omega):

        bubba_0_num = gamma0_scalar**2*(beta0_scalar-gamma0_scalar**2)/(1 + chi_p / chi_s * (g_Psi * (beta0_scalar-gamma0_scalar**2) + gamma0_scalar**2 * g_Omega))**2
        bubba_1_num = gamma1_scalar**2*(beta1_scalar-gamma1_scalar**2)/(1 + chi_p / chi_s * (g_Psi * (beta1_scalar-gamma1_scalar**2) + gamma1_scalar**2 * g_Omega))**2
        bubba_0_denom = gamma0_scalar**4/(1 + chi_p / chi_s * (g_Psi * (beta0_scalar-gamma0_scalar**2) + gamma0_scalar**2 * g_Omega))**2
        bubba_1_denom = gamma1_scalar**4/(1 + chi_p / chi_s * (g_Psi * (beta1_scalar-gamma1_scalar**2) + gamma1_scalar**2 * g_Omega))**2

        num = 1/chi_p+(chi_p/chi_s)*(weight*bubba_0_num+(1-weight)*bubba_1_num)*g_Psi**2
        denom = 1/chi_p-(chi_p/chi_s)*(weight*bubba_0_denom+(1-weight)*bubba_1_denom)*g_Omega**2

        return (g_Omega/g_Psi)**2 * num/denom

    def R_easy_scalar_for_Psi_edge(g_Psi, g_Omega):

        num0 = (beta0_scalar-gamma0_scalar**2)*(beta0_scalar-(1-omega_tilde_star(g_Psi, g_Omega))*gamma0_scalar**2)
        denom0 = (1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega)**2

        num1 = (beta1_scalar-gamma1_scalar**2)*(beta1_scalar-(1-omega_tilde_star(g_Psi, g_Omega))*gamma1_scalar**2)
        denom1 = (1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-gamma1_scalar**2) + (chi_p / chi_s) * gamma1_scalar**2 * g_Omega)**2

        return weight*num0/denom0 + (1-weight)*num1/denom1

    def R_easy_scalar_for_Omega_edge(g_Psi, g_Omega):

        num0 = gamma0_scalar**2*(beta0_scalar-(1-omega_tilde_star(g_Psi, g_Omega))*gamma0_scalar**2)
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = gamma1_scalar**2*(beta1_scalar-(1-omega_tilde_star(g_Psi, g_Omega))*gamma1_scalar**2)
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-gamma1_scalar**2) + (chi_p / chi_s) *gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1

    def R_easy_scalar_for_Omega(g_Psi, g_Omega):

        num0 = gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-gamma1_scalar**2) + (chi_p / chi_s) *gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1

    eqs = np.empty(2)

    # eqs[0] RHS (kept identical to the original; wrapped in parentheses purely
    # for readability so every term stays on the right-hand side).
    eqs[0] = ((1-1/chi_p)*1/g_Omega_star/g_Psi_star**2
              - omega_tilde_star(g_Psi_star, g_Omega_star)/chi_p/g_Psi_star**2/g_Omega_star
              + 2/chi_p/g_Psi_star**3
              - (chi_p/chi_s)*R_easy_scalar_for_Psi_edge(g_Psi_star, g_Omega_star)*1/g_Omega_star)
    eqs[1] = (1/chi_p)*g_Psi_star/g_Omega_star-delta_t * (gamma0_scalar**2*weight/(at**2*var0) + gamma1_scalar**2*(1-weight)/(at**2*var1))*g_Psi_star-(1.0/chi_p)-R_easy_scalar_for_Omega(g_Psi_star, g_Omega_star)*g_Psi_star

    return eqs

def spectral_edge_finite_centroids(g_Psi_star, g_Omega_star, diag, params):

    weight = params["weight"]
    beta0_scalar = params["beta0"]
    beta1_scalar = params["beta1"]
    gamma0_scalar = params["gamma0"]
    gamma1_scalar = params["gamma1"]

    def R_easy_scalar_for_Psi(g_Psi, g_Omega):

        num0 = beta0_scalar-gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = beta1_scalar-gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-gamma1_scalar**2) + (chi_p / chi_s) * gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1

    return diag + R_easy_scalar_for_Psi(g_Psi_star, g_Omega_star) - (1-1/chi_p)/g_Psi_star - (1.0/chi_p)*g_Omega_star/(g_Psi_star**2)

# =============================================================================
# 6. MAIN DRIVER
# -----------------------------------------------------------------------------
# The two regimes share the same structure: compute the Hermite coefficients,
# pack them into `params`, call the corresponding solver, then write the
# spectral density, (finite only) the spectral edges, and the training loss.
# =============================================================================
if __name__ == "__main__":

    output_dir = f"results_asymptotic/solver_saddle_point_equations_{mode_centroid}/P={int(d*chi_p)}_D={d}_S={int(d*chi_s)}/centroid={centroids_factor}_t={t}/"

    ## create output directory if it does not exist

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if mode_centroid == "extensive":

        start = time.time()

        print("Computing vector hermite coefficients with trapezoidal integration for extensive centroids...")

        alpha0 = alpha_trapz(mode_centroid, Gammas, at, 0)
        alpha1 = alpha_trapz(mode_centroid, Gammas, at, 1)

        beta_tilde0 = beta_tilde_trapz(mode_centroid, Gammas, at, 0)
        beta_tilde1 = beta_tilde_trapz(mode_centroid, Gammas, at, 1)

        gamma0 = gamma_trapz(mode_centroid, Gammas, at, vars, 0)
        gamma1 = gamma_trapz(mode_centroid, Gammas, at, vars, 1)

        if t > 0:

            n_points_beta_2d = npoints_integration
            n_mu_nodes_beta = npoints_integration
            z_bound_beta = 10.0

            beta0, mu_nodes_beta_0, beta_nodes_0 = beta_multiple_values(
                mu_interval, vars[0], n_points_for_mu_grid=n_mu_nodes_beta, n_points_2d=n_points_beta_2d, z_bound=z_bound_beta
            )
            beta1, mu_nodes_beta_1, beta_nodes_1 = beta_multiple_values(
                mu_interval, vars[1], n_points_for_mu_grid=n_mu_nodes_beta, n_points_2d=n_points_beta_2d, z_bound=z_bound_beta
            )

        elif t == 0:

            beta0 = beta_tilde0
            beta1 = beta_tilde1

        else:
            raise ValueError("t cannot be negative")

        end = time.time()
        print("Computed coefficients for extensive centroids! Time taken: {:.2f} seconds".format(end - start))

        params = {
        "alpha0": alpha0,
        "alpha1": alpha1,
        "beta0": beta0,
        "beta1": beta1,
        "gamma0": gamma0,
        "gamma1": gamma1,
        "beta_tilde0": beta_tilde0,
        "beta_tilde1": beta_tilde1,
        }

        start = time.time()

        print("Solving saddle point equations for extensive centroids...")

        solution = solve_equations_extensive_centroids(params)

        g_real = solution[0]
        rho = solution[1]/np.pi
        g_Omega00_real = solution[2]
        rho_Omega00 = solution[3]/np.pi
        g_Omega01_real = solution[4]
        rho_Omega01 = solution[5]/np.pi
        g_Omega11_real = solution[6]
        rho_Omega11 = solution[7]/np.pi

        end = time.time()

        print("Solved saddle point equations for extensive centroids! Time taken: {:.2f} seconds".format(end - start))

        ## print solution on file

        P = int(d*chi_p)
        S = int(d*chi_s)

        fname = f"{output_dir}/rho_lambda_analytical_weight={weight}_var1={vars[0]}_var2={vars[1]}_ortho.txt"

        #remove file if it already exists
        if os.path.exists(fname):
            os.remove(fname)

        for i in range(len(lambda_values)):
            print(f"{lambda_values[i]} {g_real[i]} {rho[i]} {g_Omega00_real[i]} {rho_Omega00[i]} {g_Omega01_real[i]} {rho_Omega01[i]} {g_Omega11_real[i]} {rho_Omega11[i]}\n", file=open(fname, "a"), end="")

        ## fully analytical training loss

        print("Computing training loss for extensive centroids...")

        start = time.time()

        threshold = (1/chi_p) / np.pi / 10

        ## sets to zero the values of rho_Omega00, rho_Omega01 and rho_Omega11 where rho is smaller than threshold (to avoid numerical issues in the integration)

        rho_Omega00_for_integration = np.array([rhoOmega*(rho>threshold) for rhoOmega, rho in zip(rho_Omega00, rho)])
        rho_Omega01_for_integration = np.array([rhoOmega*(rho>threshold) for rhoOmega, rho in zip(rho_Omega01, rho)])
        rho_Omega11_for_integration = np.array([rhoOmega*(rho>threshold) for rhoOmega, rho in zip(rho_Omega11, rho)])

        std0 = np.sqrt(vars[0])
        std1 = np.sqrt(vars[1])

        prefactor = delta_t * chi_p / at**2

        loss_train = np.zeros(training_times.size)

        for count, tau in enumerate(training_times):

            dumping_bulk = 1 - np.exp(-4*tau*lr*lambda_values)

            loss_train[count] = 1.0 - prefactor * (weight**2/vars[0]*np.trapezoid(rho_Omega00_for_integration*dumping_bulk/lambda_values, lambda_values)+(1-weight)**2/vars[1]*np.trapezoid(rho_Omega11_for_integration*dumping_bulk/lambda_values, lambda_values)+2*weight*(1-weight)/np.sqrt(vars[0]*vars[1])*np.trapezoid(rho_Omega01_for_integration*dumping_bulk/lambda_values, lambda_values))

        end = time.time()
        print("Computed training loss for extensive centroids! Time taken: {:.2f} seconds".format(end - start))

    else:
        print("Computing scalar hermite coefficients with trapezoidal integration for non-extensive centroids...")

        gamma0_scalar = gamma_trapz(mode_centroid, Gammas, at, vars, 0)
        gamma1_scalar = gamma_trapz(mode_centroid, Gammas, at, vars, 1)

        beta_tilde0_scalar = beta_tilde_trapz(mode_centroid, Gammas, at, 0)
        beta_tilde1_scalar = beta_tilde_trapz(mode_centroid, Gammas, at, 1)

        if t > 0:

            # Deterministic 2D integration with adaptive grid refinement.
            # tol=1e-5 gives a conservative target for final beta accuracy < 1e-4.
            beta0_scalar, n_beta0, err_beta0 = beta_trapz_2d(
                at, delta_t, mode_centroid, vars[0], tol=1e-5, n_init=1201, n_max=9601, z_bound=10.0
            )
            beta1_scalar, n_beta1, err_beta1 = beta_trapz_2d(
                at, delta_t, mode_centroid, vars[1], tol=1e-5, n_init=1201, n_max=9601, z_bound=10.0
            )

            print(f"beta0 deterministic grid points = {n_beta0}, refinement error est = {err_beta0:.3e}")
            print(f"beta1 deterministic grid points = {n_beta1}, refinement error est = {err_beta1:.3e}")

        elif t == 0:

            beta0_scalar = beta_tilde0_scalar
            beta1_scalar = beta_tilde1_scalar
        else:
            raise ValueError("t cannot be negative")

        print("Computed coefficients for finite centroids:")

        print("beta0 =", beta0_scalar)
        print("beta1 =", beta1_scalar)
        print("gamma0 =", gamma0_scalar)
        print("gamma1 =", gamma1_scalar)
        print("beta_tilde0 =", beta_tilde0_scalar)
        print("beta_tilde1 =", beta_tilde1_scalar)
        print("diagonal term =", D_call(weight, at, vars, [beta_tilde0_scalar, beta_tilde1_scalar], [beta0_scalar, beta1_scalar], [gamma0_scalar, gamma1_scalar]))

        params = {
        "beta0": beta0_scalar,
        "beta1": beta1_scalar,
        "gamma0": gamma0_scalar,
        "gamma1": gamma1_scalar,
        "beta_tilde0": beta_tilde0_scalar,
        "beta_tilde1": beta_tilde1_scalar,
        "delta_t": delta_t,
        "var0": vars[0],
        "var1": vars[1],
        "at": at,
        "weight": weight
        }

        print("Solving saddle point equations for finite centroids...")
        start = time.time()

        real_g, rho, real_g_Omega, rho_Omega = solve_equations_finite_centroids(params)

        end = time.time()
        print("Solved saddle point equations for finite centroids! Time taken: {:.2f} seconds".format(end - start))

        ## print solution on file

        P = int(d*chi_p)
        S = int(d*chi_s)

        thr = 1e-6

        fname = f"{output_dir}/rho_lambda_analytical_weight={weight}_var1={vars[0]}_var2={vars[1]}_ortho.txt"

        #remove file if it already exists
        if os.path.exists(fname):
            os.remove(fname)

        for i in range(len(lambda_values)):
            print(f"{lambda_values[i]} {rho[i]*(rho[i]>thr)} {rho_Omega[i]*(rho_Omega[i]>thr)}\n", file=open(fname, "a"), end="")


        ## solve for the spectral edges

        ind_lambdas_for_solver_edge = []

        '''
        for i, lambda_val in enumerate(lambda_values[:-2]):
            if np.abs(np.diff(rho)[i]) > threshold and (rho[i]< 1e-12 or rho[i+2]<1e-12):
                ind_lambdas_for_solver_edge.append(i)
        '''

        for i, lambda_val in enumerate(lambda_values[1:]):

            if (rho[i] > thr and rho[i-1]<thr) or (rho[i]<thr and rho[i-1]>thr):
                if (rho[i] > thr and rho[i-1]<thr):
                    ind_lambdas_for_solver_edge.append(i)
                else:
                    ind_lambdas_for_solver_edge.append(i-1)
                print(i, lambda_values[i], lambda_values[i-1], rho[i], rho[i-1])

        ind_lambdas_for_solver_edge = np.array(ind_lambdas_for_solver_edge)

        lambda_star = np.zeros(ind_lambdas_for_solver_edge.size)

        for count, i in enumerate(ind_lambdas_for_solver_edge):

            g_Psi_star_initial = real_g[i]
            g_Omega_star_initial = real_g_Omega[i]

            diag = D_call(weight, at, vars, [beta_tilde0_scalar, beta_tilde1_scalar], [beta0_scalar, beta1_scalar], [gamma0_scalar, gamma1_scalar])

            solution_edge = root(equations_for_spectral_edges_finite_centroids, [g_Psi_star_initial, g_Omega_star_initial], args=(params,), method='lm')

            g_Psi_star = solution_edge.x[0]
            g_Omega_star = solution_edge.x[1]

            lambda_star[count] = spectral_edge_finite_centroids(g_Psi_star, g_Omega_star, diag, params)


        lambda_star_refined = lambda_star
        real_g_for_lambda_star_refined = real_g[ind_lambdas_for_solver_edge]

        ## save lambda_star_refined on file

        fname_edges = f"{output_dir}/lambda_star_analytical_weight={weight}_var1={vars[0]}_var2={vars[1]}_ortho.txt"

        ## remove file if it already exists
        if os.path.exists(fname_edges):
            os.remove(fname_edges)

        for i in range(len(lambda_star_refined)):
            print(f"{real_g_for_lambda_star_refined[i]} {lambda_star_refined[i]}\n", file=open(fname_edges, "a"), end="")

        ## fully analytical training loss

        rho_Omega_for_integration = rho_Omega[rho_Omega>threshold]
        lambda_for_integration = lambda_values[rho_Omega>threshold]

        ## decompose lambda_for_integration in three different bulks (to detect bulk, check when difference between consecutive lambda_for_integration is larger than 10 times the previous difference)

        indices_for_bulks = []
        counter = 0
        lambda_bulks = []
        rho_Omega_bulks = []

        for i in range(1, len(lambda_for_integration)-1):
            if lambda_for_integration[i+1]-lambda_for_integration[i] > 10*(lambda_for_integration[i]-lambda_for_integration[i-1]):
                print(f"Significant change in lambda at index {i}, lambda={lambda_for_integration[i]:.6f}, diff={lambda_for_integration[i+1]-lambda_for_integration[i]:.6f}, previous diff={lambda_for_integration[i]-lambda_for_integration[i-1]:.6f}")
                indices_for_bulks.append(i)
                counter += 1
            ## save indices where this happen

        for i in range(counter+1):
            if i == 0:
                lambda_bulks.append(lambda_for_integration[:indices_for_bulks[0]+1])
                rho_Omega_bulks.append(rho_Omega_for_integration[:indices_for_bulks[0]+1])
            elif i == counter:
                lambda_bulks.append(lambda_for_integration[indices_for_bulks[-1]+1:])
                rho_Omega_bulks.append(rho_Omega_for_integration[indices_for_bulks[-1]+1:])
            else:
                lambda_bulks.append(lambda_for_integration[indices_for_bulks[i-1]+1:indices_for_bulks[i]+1])
                rho_Omega_bulks.append(rho_Omega_for_integration[indices_for_bulks[i-1]+1:indices_for_bulks[i]+1])

        std0 = np.sqrt(vars[0])
        std1 = np.sqrt(vars[1])

        coeff = (weight*gamma0_scalar/std0+(1-weight)*gamma1_scalar/std1)**2 * delta_t/at**2 * chi_p

        loss_train = np.zeros(training_times.size)

        for count, tau in enumerate(training_times):

            dumping_bulk = 1 - np.exp(-4*tau*lr*lambda_bulks[0])

            loss_train[count] = 1.0 - coeff * np.trapezoid(rho_Omega_bulks[0]*dumping_bulk/lambda_bulks[0], lambda_bulks[0])

            for i in range(1, len(lambda_bulks)):

                dumping_bulk = 1 - np.exp(-4*tau*lr*lambda_bulks[i])
                loss_train[count] -= coeff * np.trapezoid(rho_Omega_bulks[i]*dumping_bulk/lambda_bulks[i], lambda_bulks[i])

    ## save fully-analytical training loss

    ## if file exists, remove it

    fname = f"{output_dir}/training_loss_fully_analytical_weight={weight}_var1={vars[0]}_var2={vars[1]}_ortho.txt"

    if os.path.exists(fname):
        os.remove(fname)

    for count in range(training_times.size):
        print(f"{training_times[count]} {loss_train[count]}\n", file=open(fname, "a"), end="")