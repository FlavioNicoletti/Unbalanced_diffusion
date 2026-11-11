#!/usr/bin/env python3
"""
================================================================================
 Training a random-feature score model on an imbalanced Gaussian mixture
================================================================================

OUTLINE OF THE SCRIPT
---------------------
We train a *random-feature* score model by (denoising) score matching on a
two-cluster Gaussian mixture — a minimal model of *class imbalance* in the
context of diffusion generative models. For several random realizations we:

  1. Draw a synthetic dataset of `n` points in dimension `d`, split into two
     Gaussian clusters with tunable mixing weight, variances and centroid
     geometry (their cosine similarity).
  2. Build a *frozen* random projection matrix W (the "random features",
     shape p x d) and a *learnable* readout matrix A (shape d x p).
  3. Optimize A by gradient descent (plain SGD or Adam) on the denoising
     score-matching objective, using freshly sampled noise every step.
  4. At log-spaced checkpoints, evaluate and record three families of metrics:
        - the TRAIN loss   (score-matching loss on the training set),
        - the TEST loss    (same loss on fresh samples; also per cluster),
        - the SCORE ERROR  (squared distance between the learned score and the
                            *analytically known* Bayes-optimal score of the
                            mixture; also per cluster).
  5. Append all recorded curves to `.npy` files under `results_training/`.

THE MODEL
---------
The score network is a one-hidden-layer random-feature map

        S_A(x) = A * phi( W x / sqrt(d) ),

where W (p x d) is fixed at initialization and only A (d x p) is trained. The
activation phi is chosen with -activation among the entries of the ACTIVATIONS
dictionary at the top of the script (default: tanh; relu is also available).
To use another activation, add it to the dictionary.

THE DIFFUSION
-------------
A clean sample x is corrupted as  x_t = a_t * x + sqrt(delta_t) * eta,
with eta ~ N(0, I). Two noising schedules are supported:
    - "OU" (Ornstein-Uhlenbeck):  a_t = exp(-t),  delta_t = 1 - a_t^2
    - "FD" (Free Diffusion):      a_t = 1,         delta_t = t
The denoising score-matching loss trains S_A to predict -eta / sqrt(delta_t).

THE TRUE (BAYES-OPTIMAL) SCORE
------------------------------
For a Gaussian mixture the exact score of the noised density is known in closed
form. It is assembled from three "responsibility-like" quantities f, f_0, f_1
(see the helper functions), each evaluated with `logsumexp` for numerical
stability. The SCORE ERROR metric compares the learned S_A against this exact
score, so we can see how close training gets to the optimum.

--------------------------------------------------------------------------------
HOW TO RUN THE SCRIPT
--------------------------------------------------------------------------------
Basic run with defaults:

    python Training_random_features_mixture_unbalance_userfriendly.py

Custom run (dimension d=1000, chi_p=2, 3e6 iterations, Adam, 5 realizations):

    python Training_random_features_mixture_unbalance_userfriendly.py \
        -d 1000 -chi_p 2 -num_epochs 3000000 -optimizer Adam -n_realizations 5

Same run with ReLU activation:

    python Training_random_features_mixture_unbalance_userfriendly.py \
        -d 1000 -chi_p 2 -num_epochs 3000000 -optimizer Adam -n_realizations 5 -activation relu

See all options:

    python Training_random_features_mixture_unbalance_userfriendly.py --help

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    numpy, torch   (a CUDA GPU is used automatically if available, else CPU).

--------------------------------------------------------------------------------
STRUCTURE OF THE OUTPUT FOLDER
--------------------------------------------------------------------------------
A nested folder is created under `results_training/results_training_RF/` whose
name encodes the centroid factors and the run parameters (including the
activation), e.g.:

    results_training/results_training_RF/
        centroids_factor1_1.0_centroids_factor2_1.0/
            Chi_p=2.0_Chi_s=1.0_d=2e+03_t=1e-02_GD_activation=tanh/
                Train_loss_weight=0.5_var1=0.5_var2=0.5.npy
                Test_loss_weight=0.5_var1=0.5_var2=0.5.npy
                Test_loss_1_weight=...npy    (cluster 1 only)
                Test_loss_2_weight=...npy    (cluster 2 only)
                Error_score_weight=...npy
                Error_score_1_weight=...npy  (cluster 1 only)
                Error_score_2_weight=...npy  (cluster 2 only)
                Training_time_weight=...npy  (rescaled training time axis)

Each file is a 2D array with one ROW per realization (see `append_to_npy`).
================================================================================
"""

import os
import secrets
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim


# Top-level output root. Every run writes below `<main_dir>/results_training_RF/`.
# Change this single constant to redirect all results elsewhere.
main_dir = "results_training"

# Available activation functions phi of the random features (selected with -activation).
# To use another activation, add an entry here: any function acting elementwise
# on torch tensors works (e.g. torch.sigmoid, torch.nn.functional.gelu, or a lambda).
ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
}


# =============================================================================
# 1. COMMAND-LINE ARGUMENTS
# -----------------------------------------------------------------------------
# Every knob of the experiment is exposed here so you can sweep parameters from
# the shell without editing the file. Run with --help to see them all.
# =============================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="Training a random-feature diffusion score model with class imbalance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Architecture / dataset sizes (as ratios to the dimension d) ---------
    # chi_p = p/d controls the number of random features; chi_s = n/d the number
    # of training samples. Working with ratios keeps the "high-dimensional"
    # regime well defined as d grows.
    parser.add_argument("-chi_p", "--chi_p", type=float, default=2.0, help="chi_p = p/d (features per dimension)")
    parser.add_argument("-chi_s", "--chi_s", type=float, default=1.0, help="chi_s = n/d (samples per dimension)")
    parser.add_argument("-d", "--d", type=int, default=int(2e3), help="Ambient data dimension d")

    # --- Diffusion process ---------------------------------------------------
    parser.add_argument("-t", "--t", type=float, default=1e-2, help="Diffusion time t")
    parser.add_argument("-diffusion_type", "--diffusion_type", choices=["FD", "OU"], default="OU",
                        help="Noising schedule: 'FD' (Free Diffusion) or 'OU' (Ornstein-Uhlenbeck)")

    # --- Data distribution (Gaussian mixture with isotropic clusters) --------
    parser.add_argument("-weight", "--weight", type=float, default=0.5,
                        help="Mixing weight of cluster 1 (cluster 2 gets 1 - weight)")
    parser.add_argument("-var1", "--var1", type=float, default=0.5, help="Variance of cluster 1")
    parser.add_argument("-var2", "--var2", type=float, default=0.5, help="Variance of cluster 2")
    parser.add_argument("-r", "--r", type=float, default=0.0,
                        help="Cosine similarity between the two cluster centroids, in [-1, 1]")
    parser.add_argument("-centroids_factor1", "--centroids_factor1", type=float, default=1.0,
                        help="Length scale of centroid 1")
    parser.add_argument("-centroids_factor2", "--centroids_factor2", type=float, default=1.0,
                        help="Length scale of centroid 2")

    # --- Activation of the random features ------------------------------------
    parser.add_argument("-activation", "--activation", type=str.lower, default="tanh",
                        choices=list(ACTIVATIONS),
                        help="Activation function of the random features (keys of ACTIVATIONS)")

    # --- Training ------------------------------------------------------------
    parser.add_argument("-optimizer", "--optimizer", choices=["GD", "Adam"], default="GD",
                        help="Optimizer: 'GD' (plain SGD) or 'Adam'")
    parser.add_argument("-num_epochs", "--num_epochs", type=int, default=int(5e6),
                        help="Number of gradient-descent iterations")

    # --- Output --------------------------------------------------------------
    parser.add_argument("-output_dir", "--output_dir", type=str,
                        default=f"{main_dir}/results_training_RF/",
                        help="Directory to save results")

    # --- Averaging -----------------------------------------------------------
    parser.add_argument("-n_realizations", "--n_realizations", type=int, default=1,
                        help="Number of independent training realizations")

    return parser.parse_args()


# =============================================================================
# 2. DIFFUSION / DATASET CONFIGURATION
# -----------------------------------------------------------------------------
# A small container that bundles the scalar parameters shared across the model
# and the loss functions. Passing this explicitly (instead of relying on values
# captured from an enclosing scope) keeps the model self-contained and testable.
# =============================================================================
@dataclass
class DiffusionConfig:
    d: int                      # ambient dimension
    n: int                      # number of training samples
    a_t: float                  # signal-scaling coefficient of the diffusion
    delta_t: float              # noise variance of the diffusion
    weight: float               # mixing weight of cluster 1
    var_1: float                # variance of cluster 1
    var_2: float                # variance of cluster 2
    centroids_factor1: float    # length scale of centroid 1
    centroids_factor2: float    # length scale of centroid 2
    r: float                    # cosine similarity of the two centroids
    device: torch.device        # 'cuda' or 'cpu'


# =============================================================================
# 3. DATA GENERATION
# -----------------------------------------------------------------------------
# Draw a two-cluster isotropic Gaussian mixture. The two centroids are built to
# have a prescribed cosine similarity `r`: we start from the fully anti-aligned
# pair (+1..+1) and (-1..-1) and flip the sign of a fraction of the coordinates
# of the second centroid so the resulting overlap matches `r`.
# =============================================================================
def generate_dataset(n, d, weight, vars, centroids_factor1, centroids_factor2, r, device):
    n_1 = int(weight * n)   # samples in cluster 1
    n_2 = n - n_1           # samples in cluster 2

    var_1 = vars[0]
    var_2 = vars[1]

    # Spatially homogeneous centroids with the requested cosine similarity.
    m_1 = centroids_factor1 * torch.ones(d, device=device)
    m_2 = -centroids_factor2 * torch.ones(d, device=device)
    if d % 2 == 0:
        ind = int(r * d // 2)
        m_2[d // 2 - ind:] *= -1.0
    else:
        ind = int(r * (d - 1) // 2)
        m_2[d // 2 - ind:d // 2 + ind + 1] *= -1.0

    # Isotropic Gaussian samples around each centroid, standard deviation sqrt(var).
    cluster_1_samples = torch.randn(n_1, d, device=device) * np.sqrt(var_1) + m_1
    cluster_2_samples = torch.randn(n_2, d, device=device) * np.sqrt(var_2) + m_2
    dataset = torch.vstack([cluster_1_samples, cluster_2_samples])

    return [dataset, m_1, m_2]


# =============================================================================
# 4. TRUE (BAYES-OPTIMAL) SCORE — HELPER FUNCTIONS
# -----------------------------------------------------------------------------
# The exact score of the noised Gaussian mixture is
#
#     s(x) = -f({|x - m_c*a_t|^2}_c) * x
#            + ( weight*f_0(...)*m_1 + (1-weight)*f_1(...)*m_2 ) * a_t
#
# where f, f_0, f_1 are ratios of Gaussian densities ("responsibilities").
# Because these involve exponentials of large (d-dimensional) quadratic forms,
# every ratio is evaluated in LOG SPACE with `torch.logsumexp`, which avoids the
# overflow/underflow that a naive exp() of the numerator and denominator would
# cause. Here inv_Gamma_c = 1 / (var_c * a_t^2 + delta_t) is the inverse noised
# variance of cluster c, and `log_ratio` accounts for the per-cluster Gaussian
# normalization in d dimensions.
# =============================================================================
def f(ys, weight, vars, a_t, delta_t, d):
    """f = (w1*invG1*exp(E1) + w2*invG2*exp(E2)) / (w1*exp(E1) + w2*exp(E2))."""
    y_1 = ys[0]
    y_2 = ys[1]

    y_1_sq_norm = torch.linalg.norm(y_1, axis=1, keepdims=True) ** 2
    y_2_sq_norm = torch.linalg.norm(y_2, axis=1, keepdims=True) ** 2

    var_1 = vars[0]
    var_2 = vars[1]

    inv_Gamma_1 = (var_1 * a_t ** 2 + delta_t) ** (-1)
    inv_Gamma_2 = (var_2 * a_t ** 2 + delta_t) ** (-1)

    # Log-space exponents for the two mixture components.
    log_ratio = (d / 2) * torch.log(torch.tensor(inv_Gamma_2 / inv_Gamma_1, device=y_1.device))
    E_1 = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(weight, device=y_1.device))
    E_2 = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(1 - weight, device=y_1.device))

    # Denominator: w1*exp(E1) + w2*exp(E2), via logsumexp over the 2 components.
    log_weights_denom = torch.cat([E_1, E_2], dim=1)
    log_denom = torch.logsumexp(log_weights_denom, dim=1, keepdim=True)

    # Numerator: w1*invG1*exp(E1) + w2*invG2*exp(E2).
    log_num_1 = E_1 + torch.log(torch.tensor(inv_Gamma_1, device=y_1.device))
    log_num_2 = E_2 + torch.log(torch.tensor(inv_Gamma_2, device=y_1.device))
    log_weights_num = torch.cat([log_num_1, log_num_2], dim=1)
    log_num = torch.logsumexp(log_weights_num, dim=1, keepdim=True)

    return torch.exp(log_num - log_denom)


def f_0(ys, weight, vars, a_t, delta_t, d):
    """f_0 = invG1*exp(E1_raw) / (w1*exp(E1) + w2*exp(E2))  (cluster-1 term)."""
    y_1 = ys[0]
    y_2 = ys[1]

    y_1_sq_norm = torch.linalg.norm(y_1, axis=1, keepdims=True) ** 2
    y_2_sq_norm = torch.linalg.norm(y_2, axis=1, keepdims=True) ** 2

    var_1 = vars[0]
    var_2 = vars[1]

    inv_Gamma_1 = (var_1 * a_t ** 2 + delta_t) ** (-1)
    inv_Gamma_2 = (var_2 * a_t ** 2 + delta_t) ** (-1)

    log_ratio = (d / 2) * torch.log(torch.tensor(inv_Gamma_2 / inv_Gamma_1, device=y_1.device))
    E_1 = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(weight, device=y_1.device))
    E_2 = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(1 - weight, device=y_1.device))

    log_weights_denom = torch.cat([E_1, E_2], dim=1)
    log_denom = torch.logsumexp(log_weights_denom, dim=1, keepdim=True)

    # Numerator: invG1 * exp(-0.5 * |y1|^2 * invG1)   (no mixing-weight factor).
    log_num = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(inv_Gamma_1, device=y_1.device))

    return torch.exp(log_num - log_denom)


def f_1(ys, weight, vars, a_t, delta_t, d):
    """f_1 = invG2*(invG2/invG1)^d*exp(E2_raw) / (w1*exp(E1) + w2*exp(E2))  (cluster-2 term)."""
    y_1 = ys[0]
    y_2 = ys[1]

    y_1_sq_norm = torch.linalg.norm(y_1, axis=1, keepdims=True) ** 2
    y_2_sq_norm = torch.linalg.norm(y_2, axis=1, keepdims=True) ** 2

    var_1 = vars[0]
    var_2 = vars[1]

    inv_Gamma_1 = (var_1 * a_t ** 2 + delta_t) ** (-1)
    inv_Gamma_2 = (var_2 * a_t ** 2 + delta_t) ** (-1)

    log_ratio = (d / 2) * torch.log(torch.tensor(inv_Gamma_2 / inv_Gamma_1, device=y_1.device))
    E_1 = -0.5 * y_1_sq_norm * inv_Gamma_1 + torch.log(torch.tensor(weight, device=y_1.device))
    E_2 = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(1 - weight, device=y_1.device))

    log_weights_denom = torch.cat([E_1, E_2], dim=1)
    log_denom = torch.logsumexp(log_weights_denom, dim=1, keepdim=True)

    # Numerator: invG2 * (invG2/invG1)^d * exp(-0.5 * |y2|^2 * invG2)  (no weight factor).
    log_num = -0.5 * y_2_sq_norm * inv_Gamma_2 + log_ratio + torch.log(torch.tensor(inv_Gamma_2, device=y_1.device))

    return torch.exp(log_num - log_denom)


# =============================================================================
# 5. MODEL DEFINITION
# -----------------------------------------------------------------------------
# Random-feature score model  S_A(x) = A * phi(W x / sqrt(d)), with phi the chosen
# activation (an entry of ACTIVATIONS).
# W (p x d) is frozen at initialization; only A (d x p) is learned.
# The class also carries the loss / evaluation methods so that they no longer
# rely on closed-over state from main().
# =============================================================================
class ScoreModel:
    def __init__(self, W, A, cfg: DiffusionConfig, activation=torch.tanh):
        self.W = W                  # frozen random features, shape (p, d)
        self.A = A                  # learnable readout, shape (d, p)
        self.cfg = cfg
        self.activation = activation  # elementwise activation phi (see ACTIVATIONS)

    def __call__(self, x_input):
        # Forward pass: S_A(x) = A * phi(W x / sqrt(d)).
        d, device = self.cfg.d, self.cfg.device
        return (self.A @ self.activation(self.W @ x_input.T / torch.sqrt(torch.tensor(d, device=device)))).T

    # ------------------------------------------------------------------
    # Training objective (single noise draw).
    # Denoising score matching: corrupt x -> a_t*x + sqrt(delta_t)*eta and
    # push sqrt(delta_t)*S_A(x_t) towards -eta. Normalized by (d * n).
    def train_objective(self, x, eta):
        cfg = self.cfg
        S_A_x = self(cfg.a_t * x + np.sqrt(cfg.delta_t) * eta)
        l2_term = torch.sum((np.sqrt(cfg.delta_t) * S_A_x + eta) ** 2) / (cfg.d * cfg.n)
        return l2_term

    # ------------------------------------------------------------------
    # Train loss: same objective as above, averaged over `m` noise draws.
    def compute_train_loss(self, x, m=100):
        cfg = self.cfg
        total_loss = 0
        with torch.no_grad():
            for _ in range(m):
                eta = torch.randn(cfg.n, cfg.d, device=cfg.device)
                S_A_x = self(cfg.a_t * x + np.sqrt(cfg.delta_t) * eta)
                loss = torch.sum((np.sqrt(cfg.delta_t) * S_A_x + eta) ** 2) / (cfg.d * cfg.n)
                total_loss += loss.item() / m
        return total_loss

    # ------------------------------------------------------------------
    # Test loss: the score-matching loss on FRESH samples drawn from the same
    # mixture (100x the training-set size), processed in mini-batches of 100.
    def compute_test_loss(self):
        cfg = self.cfg
        batch_size = 100
        N = 100 * cfg.n
        test_dataset = generate_dataset(N, cfg.d, cfg.weight, [cfg.var_1, cfg.var_2],
                                        cfg.centroids_factor1, cfg.centroids_factor2, cfg.r, cfg.device)[0]
        # Shuffle so the two clusters are interleaved across batches.
        perm = torch.randperm(N)
        test_dataset = test_dataset[perm]
        eta = torch.randn(N, cfg.d, device=cfg.device)
        total_loss = 0.0
        num_batches = N // batch_size
        for i in range(num_batches):
            x_batch = test_dataset[i * batch_size: (i + 1) * batch_size]
            eta_batch = eta[i * batch_size: (i + 1) * batch_size]
            S_A_x = self(cfg.a_t * x_batch + np.sqrt(cfg.delta_t) * eta_batch)
            loss = torch.sum((np.sqrt(cfg.delta_t) * S_A_x + eta_batch) ** 2) / (cfg.d * N)
            total_loss += loss.item()
        return total_loss

    # ------------------------------------------------------------------
    # Test loss restricted to a SINGLE cluster (centroid `m`, variance `var`).
    # Lets us see how the model performs on the majority vs the minority cluster.
    def compute_test_loss_single_cluster(self, m, var):
        cfg = self.cfg
        batch_size = 100
        N = 10_000
        test_dataset = torch.randn(N, cfg.d, device=cfg.device) * np.sqrt(var) + m
        eta = torch.randn(N, cfg.d, device=cfg.device)
        total_loss_cluster = 0.0
        num_batches = N // batch_size
        for i in range(num_batches):
            x_batch = test_dataset[i * batch_size: (i + 1) * batch_size]
            eta_batch = eta[i * batch_size: (i + 1) * batch_size]
            S_A_x = self(cfg.a_t * x_batch + np.sqrt(cfg.delta_t) * eta_batch)
            loss = torch.sum((np.sqrt(cfg.delta_t) * S_A_x + eta_batch) ** 2) / (cfg.d * N)
            total_loss_cluster += loss.item()
        return total_loss_cluster

    # ------------------------------------------------------------------
    # Score error: squared distance between the learned score S_A and the exact
    # score of the noised mixture (assembled from f, f_0, f_1).
    # Returns the total error plus the error restricted to samples of each
    # cluster (the *_1 / *_2 quantities).
    def compute_error_score(self):
        cfg = self.cfg
        d, device, a_t, delta_t, weight = cfg.d, cfg.device, cfg.a_t, cfg.delta_t, cfg.weight
        batch_size = 100
        N = 10_000

        # `vars` here are the *noised* per-cluster variances var_c*a_t^2 + delta_t.
        vars = [cfg.var_1 * a_t ** 2 + delta_t, cfg.var_2 * a_t ** 2 + delta_t]
        # A mixed test set (drawn at the noised scale) plus one set per cluster.
        test_dataset, m_1, m_2 = generate_dataset(N, d, weight, vars,
                                                  cfg.centroids_factor1 * a_t, cfg.centroids_factor2 * a_t, cfg.r, device)
        test1 = torch.randn(N, d, device=device) * np.sqrt(cfg.var_1 * a_t ** 2 + delta_t) + m_1 * a_t
        test2 = torch.randn(N, d, device=device) * np.sqrt(cfg.var_2 * a_t ** 2 + delta_t) + m_2 * a_t

        # Shuffle the mixed set so batches interleave the two clusters.
        perm = torch.randperm(N)
        test_dataset = test_dataset[perm]

        total_loss = 0.0
        total_loss_cluster_1 = 0.0
        total_loss_cluster_2 = 0.0
        num_batches = N // batch_size
        for i in range(num_batches):
            x_batch = test_dataset[i * batch_size: (i + 1) * batch_size]
            x_batch_1 = test1[i * batch_size: (i + 1) * batch_size]
            x_batch_2 = test2[i * batch_size: (i + 1) * batch_size]

            # Learned scores on the mixed set and on each single-cluster set.
            S_A_x = self(x_batch)
            S_A_x_1 = self(x_batch_1)
            S_A_x_2 = self(x_batch_2)

            # --- Exact score on the mixed batch ---
            y1 = x_batch - torch.outer(torch.ones(batch_size, device=device), m_1)
            y2 = x_batch - torch.outer(torch.ones(batch_size, device=device), m_2)
            ys = [y1, y2]
            f_vals = f(ys, weight, vars, a_t, delta_t, d)
            f0_vals = f_0(ys, weight, vars, a_t, delta_t, d)
            f1_vals = f_1(ys, weight, vars, a_t, delta_t, d)

            # --- Exact score on the cluster-1 batch (centroids scaled by a_t) ---
            y1_1 = x_batch_1 - torch.outer(torch.ones(batch_size, device=device), m_1 * a_t)
            y1_2 = x_batch_1 - torch.outer(torch.ones(batch_size, device=device), m_2 * a_t)
            ys_1 = [y1_1, y1_2]
            f_vals_1 = f(ys_1, weight, vars, a_t, delta_t, d)
            f0_vals_1 = f_0(ys_1, weight, vars, a_t, delta_t, d)
            f1_vals_1 = f_1(ys_1, weight, vars, a_t, delta_t, d)

            # --- Exact score on the cluster-2 batch ---
            y2_1 = x_batch_2 - torch.outer(torch.ones(batch_size, device=device), m_1 * a_t)
            y2_2 = x_batch_2 - torch.outer(torch.ones(batch_size, device=device), m_2 * a_t)
            ys_2 = [y2_1, y2_2]
            f_vals_2 = f(ys_2, weight, vars, a_t, delta_t, d)
            f0_vals_2 = f_0(ys_2, weight, vars, a_t, delta_t, d)
            f1_vals_2 = f_1(ys_2, weight, vars, a_t, delta_t, d)

            # Assemble the exact scores: s = -f*x + (w*f0*m1 + (1-w)*f1*m2)*a_t.
            true_score = -f_vals * x_batch + (weight * f0_vals * torch.outer(torch.ones(batch_size, device=device), m_1)
                                              + (1 - weight) * f1_vals * torch.outer(torch.ones(batch_size, device=device), m_2))
            true_score_1 = -f_vals_1 * x_batch_1 + (weight * f0_vals_1 * torch.outer(torch.ones(batch_size, device=device), m_1 * a_t)
                                                    + (1 - weight) * f1_vals_1 * torch.outer(torch.ones(batch_size, device=device), m_2 * a_t))
            true_score_2 = -f_vals_2 * x_batch_2 + (weight * f0_vals_2 * torch.outer(torch.ones(batch_size, device=device), m_1 * a_t)
                                                    + (1 - weight) * f1_vals_2 * torch.outer(torch.ones(batch_size, device=device), m_2 * a_t))

            # Squared errors, normalized by (d * N).
            loss = torch.sum((S_A_x - true_score) ** 2) / (d * N)
            loss_cluster_1 = torch.sum((S_A_x_1 - true_score_1) ** 2) / (d * N)
            loss_cluster_2 = torch.sum((S_A_x_2 - true_score_2) ** 2) / (d * N)
            total_loss += loss.item()
            total_loss_cluster_1 += loss_cluster_1.item()
            total_loss_cluster_2 += loss_cluster_2.item()
        return total_loss, total_loss_cluster_1, total_loss_cluster_2


# =============================================================================
# 6. UTILITIES
# =============================================================================
def sample_logarithmically(N, T):
    """Return up to N integer checkpoints spaced logarithmically in [1, T].

    Training is logged at these epochs so that the (typically fast) early
    dynamics and the (slow) late dynamics are both well resolved without
    logging every single step."""
    log_samples = np.linspace(np.log(1), np.log(T), N)
    samples = np.exp(log_samples)
    return sorted({int(v) for v in samples})


def append_to_npy(filepath, new_data):
    """Append `new_data` as a new ROW to an existing .npy array (or create it).

    Each realization contributes one row, so re-running with the same parameters
    accumulates rows rather than overwriting previous results."""
    new_data = np.array(new_data).reshape(1, -1)  # shape: (1, n_checkpoints)
    if os.path.exists(filepath):
        existing = np.load(filepath)
        combined = np.vstack([existing, new_data])
    else:
        combined = new_data
    np.save(filepath, combined)


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    args = get_args()

    # Seed from OS entropy so runs differ by default. The seed is printed below;
    # to make a run fully reproducible, replace this with a fixed integer, or use the seed from a past run, copying it from the related log file.
    seed = secrets.randbits(64)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Derived parameters --------------------------------------------------
    Chi_p, Chi_s, d, t = args.chi_p, args.chi_s, args.d, args.t
    diffusion_type = args.diffusion_type
    if diffusion_type == "FD":       # Free Diffusion: signal kept, noise grows linearly.
        a_t = 1.0
        delta_t = t
    elif diffusion_type == "OU":     # Ornstein-Uhlenbeck: signal decays, variance -> 1.
        a_t = np.exp(-t)
        delta_t = 1 - a_t ** 2
    p, n = int(Chi_p * d), int(Chi_s * d)
    weight, r = args.weight, args.r
    var_1 = args.var1
    var_2 = args.var2
    centroids_factor1 = args.centroids_factor1
    centroids_factor2 = args.centroids_factor2
    activation = args.activation
    phi = ACTIVATIONS[activation]

    cfg = DiffusionConfig(
        d=d, n=n, a_t=a_t, delta_t=delta_t,
        weight=weight, var_1=var_1, var_2=var_2,
        centroids_factor1=centroids_factor1, centroids_factor2=centroids_factor2,
        r=r, device=device,
    )

    # Output folder encodes the centroid factors; created if missing.
    end_dir = f"centroids_factor1_{centroids_factor1}_centroids_factor2_{centroids_factor2}"
    output_dir_full = os.path.join(args.output_dir, end_dir)
    os.makedirs(output_dir_full, exist_ok=True)

    # Learning rate is scaled by d/delta_t: the gradient is very small at
    # initialization (A starts at zero), and this scaling keeps the effective
    # step size comparable across dimensions and diffusion times.
    learning_rate = (5e-5) * d / delta_t

    print(f"\n🚀 Training random feature diffusion model on imbalanced data")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Parameters: χ_p={Chi_p}, χ_s={Chi_s}, d={d}, t={t}, activation={activation}")
    print(f"Optimizer: {args.optimizer}, lr={learning_rate:.2e}")
    print(f"Number of epochs: {args.num_epochs:,}")
    print(f"Number of realizations: {args.n_realizations}\n")

    # Log-spaced checkpoints (600 of them) at which metrics are evaluated.
    sample_times = sample_logarithmically(600, args.num_epochs)

    # Full run folder, encoding the main run parameters (and the activation) in its name.
    # Results are appended realization by realization, so runs with different
    # activations must not share a folder.
    run_name = f"Chi_p={Chi_p:.1f}_Chi_s={Chi_s:.1f}_d={d:.0e}_t={t:.0e}_{args.optimizer}_activation={activation}"
    save_dir = os.path.join(output_dir_full, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}\n")

    # --- Realization loop ----------------------------------------------------
    for realization_idx in range(args.n_realizations):
        print(f"\n{'=' * 80}")
        print(f"[Realization {realization_idx + 1}/{args.n_realizations}]")
        print(f"{'=' * 80}")

        # Fresh random features W (frozen) and readout A (learned, init at zero).
        W = torch.randn(p, d, device=device)
        A = torch.zeros(d, p, requires_grad=True, device=device)
        score_model = ScoreModel(W, A, cfg, activation=phi)

        # Fresh training dataset for this realization.
        x, m_1, m_2 = generate_dataset(n, d, weight, [var_1, var_2], centroids_factor1, centroids_factor2, r, device)

        # Optimizer over the single learnable tensor A.
        optimizer = optim.SGD([A], lr=learning_rate) if args.optimizer == "GD" else optim.Adam([A], lr=learning_rate)

        # Metric curves accumulated over this realization's checkpoints.
        Train_loss, Test_loss, Test_loss_1, Test_loss_2 = [], [], [], []
        Error_score, Error_score_1, Error_score_2, Training_time = [], [], [], []

        for epoch in range(args.num_epochs + 1):
            # One gradient step on the denoising objective with fresh noise.
            eta = torch.randn(n, d, device=device)
            loss = score_model.train_objective(x, eta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate and record metrics at the log-spaced checkpoints.
            if epoch in sample_times:
                with torch.no_grad():
                    train = score_model.compute_train_loss(x)
                    test = score_model.compute_test_loss()
                    test_1 = score_model.compute_test_loss_single_cluster(m_1, var_1)
                    test_2 = score_model.compute_test_loss_single_cluster(m_2, var_2)
                    error, error_1, error_2 = score_model.compute_error_score()

                    Train_loss.append(train)
                    Test_loss.append(test)
                    Test_loss_1.append(test_1)
                    Test_loss_2.append(test_2)
                    Error_score.append(error)
                    Error_score_1.append(error_1)
                    Error_score_2.append(error_2)
                    # Rescaled "physical" training time = epochs * learning rate.
                    Training_time.append(epoch * learning_rate)

                print(f"[Epoch {epoch:,}] Train={train:.4e} | Test={test:.4e} | "
                      f"Error on the score={error:.4e} | Error on cluster 1={error_1:.4e} | "
                      f"Error on cluster 2={error_2:.4e}")

        # Append this realization's curves (one row each) to the .npy files.
        suffix = f"weight={weight}_var1={var_1}_var2={var_2}.npy"
        append_to_npy(os.path.join(save_dir, f"Train_loss_{suffix}"), Train_loss)
        append_to_npy(os.path.join(save_dir, f"Test_loss_{suffix}"), Test_loss)
        append_to_npy(os.path.join(save_dir, f"Test_loss_1_{suffix}"), Test_loss_1)
        append_to_npy(os.path.join(save_dir, f"Test_loss_2_{suffix}"), Test_loss_2)
        append_to_npy(os.path.join(save_dir, f"Error_score_{suffix}"), Error_score)
        append_to_npy(os.path.join(save_dir, f"Training_time_{suffix}"), Training_time)
        append_to_npy(os.path.join(save_dir, f"Error_score_1_{suffix}"), Error_score_1)
        append_to_npy(os.path.join(save_dir, f"Error_score_2_{suffix}"), Error_score_2)

        print(f"\n✅ Realization {realization_idx + 1} complete. Results appended.")

        # Re-seed for the next realization so it draws independent randomness.
        realization_seed = secrets.randbits(64)
        torch.manual_seed(realization_seed)

    print(f"\n{'=' * 80}")
    print(f"✅ All {args.n_realizations} realizations complete. Results saved to: {save_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
