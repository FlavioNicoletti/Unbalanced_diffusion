# Asymptotic (self-consistent) spectrum of the diffusion feature matrix

This repository computes the **asymptotic spectral density** of the
random-feature covariance matrix `U` of a two-cluster Gaussian mixture — a simple
model of *class imbalance* in diffusion generative models — by solving the
**self-consistent (saddle-point) equations** of random-matrix theory, rather than
by diagonalizing sampled matrices. From the spectrum it also derives the
**fully-analytical training-loss curve**.

The fully-commented code is in the file
[`saddle_point_spectrum.py`](saddle_point_spectrum.py).

---

## The two centroid regimes

The mixture centroids can be in one of two regimes, selected **automatically**
from `centroids_factor` and the dimension `d` (the threshold is
`centroids_factor > sqrt(10 / d)`):

- **`extensive`** — the centroid norm grows with `d`. The Hermite coefficients
  become `p`-dependent *vectors*, characterised in the asymptotic theory as functions of $\mu$, and the saddle-point equations involve `2×2`
  matrices (`g_gamma`, `g_omega`). Solved by `solve_equations_extensive_centroids`.
- **`finite`** — the centroid norm is `O(1)`. The Hermite coefficients are
  *scalars* and the equations reduce to two scalar unknowns (`g_Psi`, `g_Omega`).
  Solved by `solve_equations_finite_centroids`.

Both solvers share the same structure, so the two branches of the main driver
are symmetric: compute the Hermite coefficients → pack them into `params` → call
the regime's solver → write the density, (finite only) the spectral edges, and
the training loss.

---

## What the script does

For the selected regime it:

1. **Computes the Hermite coefficients** (`alpha`, `beta_tilde`, `gamma`, `beta`)
   that describe how `tanh` acts on Gaussian inputs, by trapezoidal integration
   (a deterministic, adaptively-refined 2D quadrature for `beta`).
2. **Solves the self-consistent equations** for the resolvent `g(z)` along a grid
   of eigenvalues `lambda`, tracking the solution from a large imaginary part of
   `z` down to the real axis (homotopy continuation: each solve seeds the next).
   The spectral density is `rho(lambda) = |Im g(lambda)| / pi`.
3. **(finite regime)** Locates the spectral edges `lambda_star`.
4. **Integrates** the resulting density into the analytical training-loss curve.
5. **Writes** the density, the edges, and the loss curve to text files under
   `results_asymptotic/`.

---

## Installation

**Option A — pip** (into your current/virtual environment):

```bash
pip install -r requirements.txt
```

**Option B — conda** (creates an isolated environment):

```bash
conda env create -f environment.yml
conda activate class-imbalance-diffusion-asymptotic
```

Dependencies: `numpy`, `scipy` (`matplotlib` only if you re-enable the optional,
commented-out plotting blocks).

---

## Usage

Run with defaults (finite-centroid regime):

```bash
python saddle_point_spectrum.py
```

Trigger the extensive regime with a larger centroid factor:

```bash
python saddle_point_spectrum.py -d 100 -chi_p 60 -chi_s 30 -centroids_factor 1.0 -t 0.01
```

See all options:

```bash
python saddle_point_spectrum.py --help
```

---

## Command-line options

| Flag | Default | Meaning |
|------|---------|---------|
| `-d` | `100` | Ambient data dimension `d`. |
| `-chi_p` | `60.0` | Ratio `p/d` (features per dimension); the number of features is `P = d·chi_p`. |
| `-chi_s` | `30.0` | Ratio `n/d` (samples per dimension); the number of samples is `S = d·chi_s`. |
| `-weight` | `0.5` | Mixing weight of cluster 1 (cluster 2 gets `1 - weight`); the imbalance knob. |
| `-centroids_factor` | `0.1` | Centroid length scale. Also selects the regime: `> sqrt(10/d)` → extensive, else finite. |
| `-var0` | `0.5` | Variance of cluster 1. |
| `-var1` | `0.25` | Variance of cluster 2. |
| `-t` | `0.01` | Diffusion time `t` (Ornstein–Uhlenbeck: `a_t = e^{-t}`, `Δ_t = 1 - a_t²`). |
| `-npoints_integration` | `5000` | Number of points in the numerical-integration grid. |

A few parameters are fixed near the top of the script (edit there if needed):
the eigenvalue grid (`n_values`, `lambda_values`), the continuation protocol
(`n_iter` / `epsilon_protocol`), and the training-loss time grid
(`training_times`, learning rate `lr`).

---

## Output

A nested folder is created under `results_asymptotic/` whose name encodes the
regime and the main parameters, e.g. (finite regime, defaults):

```
results_asymptotic/solver_saddle_point_equations_finite/P=6000_D=100_S=3000/centroid=0.1_t=0.01/
```

Inside you get:

| File | Contents |
|------|----------|
| `rho_lambda_analytical_*.txt` | The spectral density. Finite regime: `lambda  rho(lambda)  rho_Omega(lambda)`. Extensive regime: `lambda` followed by the resolvent real part and the `rho` / `g_Omega` components (`00`, `01`, `11`). |
| `lambda_star_analytical_*.txt` | Spectral edges: `Re g(lambda_star)  lambda_star` (**finite regime only**). |
| `training_loss_fully_analytical_*.txt` | The analytical training-loss curve: `training_time  loss`. |

> **Note:** files are removed and rewritten on each run (they are not appended
> to), so re-running with the same parameters overwrites the previous output.

---

## Reproducibility

The solver is **deterministic** — the results depend only on the command-line
parameters, not on any random seed (the `seed`/`rng` defined at the top are not
used by the computation). Running twice with the same arguments reproduces the
output exactly.

---

## Performance notes

- The **extensive** regime is the expensive one: each `root` call solves a
  16-variable system whose residual involves 2D integrals over the
  `npoints_integration` grid, repeated across the `epsilon_protocol` continuation
  and the `lambda` grid. Reduce `-npoints_integration` and/or `n_values` for
  quick tests.
- The **finite** regime solves a 4-variable system per `lambda`; its cost is
  dominated by `n_values` (default `10000`) times the continuation length
  (`n_iter`). Reduce `n_values` for a quick smoke test.
- The one-off Hermite `beta` integral uses adaptive grid refinement and reports
  its estimated error at startup.
