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

The script treats the **subextensive-centroid regime**: the centroid norm is o(N), so
at leading order the centroids drop out of the asymptotic spectrum. The Hermite
coefficients are *scalars* and the saddle-point equations reduce to two scalar
unknowns (`g_Psi`, `g_Omega`), solved by `solve_equations_finite_centroids`.
For this reason the centroids are not an input of the script.

The activation of the random features is chosen with `-activation`
(`tanh` or `relu`). It only affects the Hermite coefficients, which are
evaluated numerically, so any other activation can be used by adding it to the
`ACTIVATIONS` dictionary at the top of the script. For ReLU, the in-code flag
`use_exact_relu_coefficients = True` replaces the numerical integrals with
closed-form expressions.

The script:

  1. Computes the Hermite coefficients (alpha, beta_tilde, gamma, beta) that
     describe how the activation acts on Gaussian inputs, by trapezoidal
     integration (or exactly, for ReLU). alpha = 0 for odd activations such as
     tanh, but not for ReLU.
  2. Solves the self-consistent equations for the resolvent `g(z)` along a grid
     of eigenvalues `lambda`, tracking the solution from large imaginary part
     down to the real axis (homotopy continuation in the small imaginary part).
     The spectral density is `rho(lambda) = Im g(lambda) / pi`.
  3. Locates the spectral edges `lambda_star`.
  4. Integrates the resulting density into the analytical training-loss curve.
  5. Writes the density, the edges, and the loss curve to text files under
     `results_asymptotic/`.

--------------------------------------------------------------------------------
HOW TO RUN THE SCRIPT
--------------------------------------------------------------------------------
Basic run with defaults:

    python saddle_point_spectrum.py

Custom run:

    python saddle_point_spectrum.py -d 200 -chi_p 60 -chi_s 30 -weight 0.3 -t 0.1 -activation relu

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
main parameters, e.g.:

    results_asymptotic/solver_saddle_point_equations_finite/P=6000_D=100_S=3000/t=0.01_activation=tanh/
        rho_lambda_analytical_weight=..._ortho.txt          (lambda, rho, rho_Omega)
        lambda_star_analytical_weight=..._ortho.txt         (Re g at the edge, spectral edge)
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
# Here input parameters are defined. In the command line, use --help to see them all.
# =============================================================================

## available activation functions (selected with -activation)
## To use another activation, add an entry here: its Hermite coefficients are
## evaluated numerically, so nothing else needs to change.

ACTIVATIONS = {
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x: np.maximum(x, 0.0),
}

## ReLU only: if True, use the closed-form Hermite coefficients
## (see `relu_exact_coefficients`) instead of numerical integration.

use_exact_relu_coefficients = True

parser = argparse.ArgumentParser(
    description='Solve the self-consistent saddle-point equations for the spectrum of U.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-d', type=int, default=100, help='Dimensionality of the data.')
parser.add_argument('-chi_p', type=float, default=60.0, help='Value of chi_p: number of hidden neurons / dimensionality of the data.')
parser.add_argument('-chi_s', type=float, default=30.0, help='Value of chi_s: number of samples / dimensionality of the data.')
parser.add_argument('-weight', type=float, default=0.5, help='Weight for the mixture of clusters: class 1 has this input weight, class 2 has the complementary weight.')
parser.add_argument('-var0', type=float, default=0.5, help='Variance of the first class.')
parser.add_argument('-var1', type=float, default=0.25, help='Variance of the second class.')
parser.add_argument('-t', type=float, default=0.01, help='Diffusion time: measures the level of diffusion noise in the Denoising Score Matching objective.')
parser.add_argument('-npoints_integration', type=int, default=5000, help='Number of integration points (for Hermite coefficients).')
parser.add_argument('-activation', type=str.lower, default='tanh', choices=list(ACTIVATIONS), help='Activation function of the random features.')

args = parser.parse_args()


# =============================================================================
# 2. DERIVED PARAMETERS AND CONFIGURATION
# -----------------------------------------------------------------------------
# Diffusion coefficients, the numerical-integration grid, and the solver /
# training-loss settings.
# =============================================================================

## input parameters

d = args.d
chi_p = args.chi_p
chi_s = args.chi_s
weight = args.weight
vars = np.array([args.var0, args.var1])
t = args.t
activation = args.activation

if t < 0:
    raise ValueError("t cannot be negative")

## diffusion parameters

at = np.exp(-t) ## dumping factor
delta_t = 1-at**2 ## variance of the diffusion noise at time t
Gammas = np.sqrt(vars*at**2+delta_t) ## variances at diffusion time t

## parameters for numerical integration

npoints_integration = args.npoints_integration
lower_bound_integration = -10.0
upper_bound_integration = 10.0
xi = np.linspace(lower_bound_integration, upper_bound_integration, npoints_integration)

## activation function

phi = ACTIVATIONS[activation]

## parameters for solver (change accordingly)

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
# Gaussian expectations of the activation phi(...) that characterize the
# non-linearity. In the finite-centroid regime each one is a SCALAR per cluster.
# They are evaluated numerically for any phi; for ReLU the closed-form expressions
# in `relu_exact_coefficients` can be used instead.
# For odd activations (e.g. tanh) alpha = E[phi(std*xi)] vanishes by symmetry;
# for ReLU it does not.
# =============================================================================

def alpha_trapz(Gammas, cluster_index):

    std = Gammas[cluster_index]

    inv_2pi_sqrt = 1/np.sqrt(2*np.pi)

    gaussian_weigth = np.exp(-0.5*xi**2)

    return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*phi(std*xi), xi)

def beta_tilde_trapz(Gammas, cluster_index):

    std = Gammas[cluster_index]

    inv_2pi_sqrt = 1/np.sqrt(2*np.pi)

    gaussian_weigth = np.exp(-0.5*xi**2)

    return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*phi(std*xi)**2, xi)

def gamma_trapz(Gammas, at, vars, cluster_index):

    std = Gammas[cluster_index]

    inv_2pi_sqrt = 1/np.sqrt(2*np.pi)

    gaussian_weigth = np.exp(-0.5*xi**2)

    return np.trapezoid(gaussian_weigth*inv_2pi_sqrt*xi*phi(std*xi), xi)*np.sqrt(vars[cluster_index])*at/Gammas[cluster_index]

def beta_trapz_2d(at, delta_t, var, tol=1e-5, n_init=1201, n_max=9601, z_bound=10.0):
    """
    Deterministic computation of beta using nested trapezoidal integration.

    beta = E_{z1}[ (E_{z2}[phi(s1*z1 + s2*z2)])^2 ],
    with s1 = sqrt(var)*at, s2 = sqrt(delta_t).

    The grid is refined until two consecutive estimates differ by < tol.
    """

    if n_init < 3:
        n_init = 3
    if n_init % 2 == 0:
        n_init += 1

    sqrt_2pi_inv = 1.0 / np.sqrt(2.0 * np.pi)
    s1 = np.sqrt(var) * at
    s2 = np.sqrt(delta_t)

    def _beta_on_grid(n_points):
        z = np.linspace(-z_bound, z_bound, n_points)
        pdf = sqrt_2pi_inv * np.exp(-0.5 * z**2)

        # Inner expectation over z2 for each z1 (vectorized on the 2D mesh).
        arg = s1 * z[:, np.newaxis] + s2 * z[np.newaxis, :]
        inner = np.trapezoid(pdf[np.newaxis, :] * phi(arg), z, axis=1)

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

def relu_exact_coefficients(Gammas, at, delta_t, vars, cluster_index):
    """
    Closed-form Hermite coefficients for phi = ReLU, with h ~ N(0, s^2),
    s = Gammas[cluster_index]:

        alpha      = E[ReLU(h)]                 = s / sqrt(2 pi)
        beta_tilde = E[ReLU(h)^2]               = s^2 / 2
        gamma      = sqrt(var) at E[ReLU'(h)]   = sqrt(var) at / 2
        beta       = E[ReLU(u) ReLU(v)]         = s^2 / (2 pi) * (sqrt(1 - c^2) + (pi - arccos c) c)

    where u, v are N(0, s^2) with correlation c = var at^2 / s^2 (they share the
    data component s1*z1 and have independent noise components s2*z2); this is
    the first-order arc-cosine kernel. At t = 0, c = 1 and beta = beta_tilde.
    """

    s = Gammas[cluster_index]
    var = vars[cluster_index]

    alpha = s / np.sqrt(2*np.pi)
    beta_tilde = s**2 / 2
    gamma = np.sqrt(var) * at / 2

    c = np.clip(var * at**2 / s**2, -1.0, 1.0)
    beta = s**2 / (2*np.pi) * (np.sqrt(1 - c**2) + (np.pi - np.arccos(c)) * c)

    return alpha, beta_tilde, gamma, beta

def D_call(weight, at, vars, beta_tildes, betas, gammas):

    #purely diagonal term in the GEP expression of U

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
# 4. FINITE-CENTROID SADDLE-POINT EQUATIONS
# -----------------------------------------------------------------------------
# The centroid norm is o(N). The first Hermite coefficient alpha enters only
# through the centred second moment beta - alpha^2 (it vanishes for odd
# activations such as tanh); D_call is unaffected, since it depends on
# beta_tilde - beta. The self-consistent equations reduce
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

    alpha0_scalar = params["alpha0"]
    alpha1_scalar = params["alpha1"]

    gamma0_scalar = params["gamma0"]
    gamma1_scalar = params["gamma1"]

    beta_tilde0_scalar = params["beta_tilde0"]
    beta_tilde1_scalar = params["beta_tilde1"]

    D_call_scalar = D_call(weight, at, vars, [beta_tilde0_scalar, beta_tilde1_scalar], [beta0_scalar, beta1_scalar], [gamma0_scalar, gamma1_scalar])

    def R_easy_scalar_for_Psi(g_Psi, g_Omega):

        num0 = beta0_scalar-alpha0_scalar**2-gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = beta1_scalar-alpha1_scalar**2-gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + (chi_p / chi_s) * gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1
    
    def R_easy_scalar_for_Omega(g_Psi, g_Omega):

        num0 = gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + (chi_p / chi_s) *gamma1_scalar**2 * g_Omega

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

    For each lambda it follows the resolvent from a large imaginary part of z down
    to the real axis (homotopy continuation: the previous solution seeds the next
    `root` call), so the branch is tracked continuously.

    Returns four arrays of length `n_values`, aligned with `lambda_values`
    (entry k is the solution at lambda_values[k]):
        real_g, rho, real_g_Omega, rho_Omega
    where rho = |Im g| / pi and rho_Omega = |Im g_Omega| / pi.
    """
    real_g = np.zeros(n_values)
    rho = np.zeros(n_values)
    real_g_Omega = np.zeros(n_values)
    rho_Omega = np.zeros(n_values)

    for count, z_real in enumerate(lambda_values):

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

    alpha0_scalar = params["alpha0"]
    alpha1_scalar = params["alpha1"]

    gamma0_scalar = params["gamma0"]
    gamma1_scalar = params["gamma1"]

    beta_tilde0_scalar = params["beta_tilde0"]
    beta_tilde1_scalar = params["beta_tilde1"]

    #diag = D_call(weight, at, vars, [beta_tilde0_scalar, beta_tilde1_scalar], [beta0_scalar, beta1_scalar], [gamma0_scalar, gamma1_scalar])

    def omega_tilde_star(g_Psi, g_Omega):

        bubba_0_num = gamma0_scalar**2*(beta0_scalar-alpha0_scalar**2-gamma0_scalar**2)/(1 + chi_p / chi_s * (g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + gamma0_scalar**2 * g_Omega))**2
        bubba_1_num = gamma1_scalar**2*(beta1_scalar-alpha1_scalar**2-gamma1_scalar**2)/(1 + chi_p / chi_s * (g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + gamma1_scalar**2 * g_Omega))**2
        bubba_0_denom = gamma0_scalar**4/(1 + chi_p / chi_s * (g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + gamma0_scalar**2 * g_Omega))**2
        bubba_1_denom = gamma1_scalar**4/(1 + chi_p / chi_s * (g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + gamma1_scalar**2 * g_Omega))**2

        num = 1/chi_p+(chi_p/chi_s)*(weight*bubba_0_num+(1-weight)*bubba_1_num)*g_Psi**2
        denom = 1/chi_p-(chi_p/chi_s)*(weight*bubba_0_denom+(1-weight)*bubba_1_denom)*g_Omega**2

        return (g_Omega/g_Psi)**2 * num/denom

    def R_easy_scalar_for_Psi_edge(g_Psi, g_Omega):

        num0 = (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2)*(beta0_scalar-alpha0_scalar**2-(1-omega_tilde_star(g_Psi, g_Omega))*gamma0_scalar**2)
        denom0 = (1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega)**2

        num1 = (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2)*(beta1_scalar-alpha1_scalar**2-(1-omega_tilde_star(g_Psi, g_Omega))*gamma1_scalar**2)
        denom1 = (1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + (chi_p / chi_s) * gamma1_scalar**2 * g_Omega)**2

        return weight*num0/denom0 + (1-weight)*num1/denom1

    def R_easy_scalar_for_Omega_edge(g_Psi, g_Omega):

        num0 = gamma0_scalar**2*(beta0_scalar-alpha0_scalar**2-(1-omega_tilde_star(g_Psi, g_Omega))*gamma0_scalar**2)
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = gamma1_scalar**2*(beta1_scalar-alpha1_scalar**2-(1-omega_tilde_star(g_Psi, g_Omega))*gamma1_scalar**2)
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + (chi_p / chi_s) *gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1

    def R_easy_scalar_for_Omega(g_Psi, g_Omega):

        num0 = gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + (chi_p / chi_s) *gamma1_scalar**2 * g_Omega

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

    alpha0_scalar = params["alpha0"]
    alpha1_scalar = params["alpha1"]
    gamma0_scalar = params["gamma0"]
    gamma1_scalar = params["gamma1"]

    def R_easy_scalar_for_Psi(g_Psi, g_Omega):

        num0 = beta0_scalar-alpha0_scalar**2-gamma0_scalar**2
        denom0 = 1 + (chi_p / chi_s) * g_Psi * (beta0_scalar-alpha0_scalar**2-gamma0_scalar**2) + (chi_p / chi_s) * gamma0_scalar**2 * g_Omega

        num1 = beta1_scalar-alpha1_scalar**2-gamma1_scalar**2
        denom1 = 1 + (chi_p / chi_s) * g_Psi * (beta1_scalar-alpha1_scalar**2-gamma1_scalar**2) + (chi_p / chi_s) * gamma1_scalar**2 * g_Omega

        return weight*num0/denom0 + (1-weight)*num1/denom1

    return diag + R_easy_scalar_for_Psi(g_Psi_star, g_Omega_star) - (1-1/chi_p)/g_Psi_star - (1.0/chi_p)*g_Omega_star/(g_Psi_star**2)

# =============================================================================
# 5. MAIN DRIVER
# -----------------------------------------------------------------------------
# Compute the Hermite coefficients, pack them into `params`, call the solver,
# then write the spectral density, the spectral edges, and the training loss.
# =============================================================================
if __name__ == "__main__":

    output_dir = f"results_asymptotic/solver_saddle_point_equations_finite/P={int(d*chi_p)}_D={d}_S={int(d*chi_s)}/t={t}_activation={activation}/"

    ## create output directory if it does not exist

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if activation == 'relu' and use_exact_relu_coefficients:

        print("Computing scalar hermite coefficients with exact ReLU expressions for finite centroids...")

        alpha0_scalar, beta_tilde0_scalar, gamma0_scalar, beta0_scalar = relu_exact_coefficients(Gammas, at, delta_t, vars, 0)
        alpha1_scalar, beta_tilde1_scalar, gamma1_scalar, beta1_scalar = relu_exact_coefficients(Gammas, at, delta_t, vars, 1)

    else:

        print(f"Computing scalar hermite coefficients ({activation}) with trapezoidal integration for finite centroids...")

        alpha0_scalar = alpha_trapz(Gammas, 0)
        alpha1_scalar = alpha_trapz(Gammas, 1)

        gamma0_scalar = gamma_trapz(Gammas, at, vars, 0)
        gamma1_scalar = gamma_trapz(Gammas, at, vars, 1)

        beta_tilde0_scalar = beta_tilde_trapz(Gammas, 0)
        beta_tilde1_scalar = beta_tilde_trapz(Gammas, 1)

        if t > 0:

            # Deterministic 2D integration with adaptive grid refinement.
            # tol=1e-5 gives a conservative target for final beta accuracy < 1e-4.
            beta0_scalar, n_beta0, err_beta0 = beta_trapz_2d(
                at, delta_t, vars[0], tol=1e-5, n_init=1201, n_max=9601, z_bound=10.0
            )
            beta1_scalar, n_beta1, err_beta1 = beta_trapz_2d(
                at, delta_t, vars[1], tol=1e-5, n_init=1201, n_max=9601, z_bound=10.0
            )

            print(f"beta0 deterministic grid points = {n_beta0}, refinement error est = {err_beta0:.3e}")
            print(f"beta1 deterministic grid points = {n_beta1}, refinement error est = {err_beta1:.3e}")

        else:

            # t == 0: no diffusion noise, so beta coincides with beta_tilde.
            beta0_scalar = beta_tilde0_scalar
            beta1_scalar = beta_tilde1_scalar

    print("Computed coefficients for finite centroids:")

    print("alpha0 =", alpha0_scalar)
    print("alpha1 =", alpha1_scalar)
    print("beta0 =", beta0_scalar)
    print("beta1 =", beta1_scalar)
    print("gamma0 =", gamma0_scalar)
    print("gamma1 =", gamma1_scalar)
    print("beta_tilde0 =", beta_tilde0_scalar)
    print("beta_tilde1 =", beta_tilde1_scalar)
    print("diagonal term =", D_call(weight, at, vars, [beta_tilde0_scalar, beta_tilde1_scalar], [beta0_scalar, beta1_scalar], [gamma0_scalar, gamma1_scalar]))

    params = {
    "alpha0": alpha0_scalar,
    "alpha1": alpha1_scalar,
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

    ## an edge is where rho crosses thr between consecutive grid points i-1 and i;
    ## store the index of the point that lies inside the support

    for i in range(1, n_values):

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
    ## loss(tau) = 1 - coeff * \int d lambda rho_Omega(lambda) (1 - exp(-4 tau lr lambda)) / lambda,
    ## integrated over the whole lambda grid. rho_Omega is set to zero below `threshold`
    ## (numerical noise outside the support), so gaps between bulks contribute nothing.

    rho_Omega_for_integration = rho_Omega * (rho_Omega > threshold)

    std0 = np.sqrt(vars[0])
    std1 = np.sqrt(vars[1])

    coeff = (weight*gamma0_scalar/std0+(1-weight)*gamma1_scalar/std1)**2 * delta_t/at**2 * chi_p

    loss_train = np.zeros(training_times.size)

    for count, tau in enumerate(training_times):

        damping = 1 - np.exp(-4*tau*lr*lambda_values)

        loss_train[count] = 1.0 - coeff * np.trapezoid(rho_Omega_for_integration*damping/lambda_values, lambda_values)

    ## save fully-analytical training loss

    ## if file exists, remove it

    fname = f"{output_dir}/training_loss_fully_analytical_weight={weight}_var1={vars[0]}_var2={vars[1]}_ortho.txt"

    if os.path.exists(fname):
        os.remove(fname)

    for count in range(training_times.size):
        print(f"{training_times[count]} {loss_train[count]}\n", file=open(fname, "a"), end="")