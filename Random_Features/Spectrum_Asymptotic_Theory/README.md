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

## The finite-centroid regime

The script treats the regime where the centroid norm is subextensive---i.e. `o(N)`, for data with dimension `N`. At leading order
the centroids drop out of the asymptotic spectrum: the Hermite coefficients are
*scalars* and the saddle-point equations reduce to two scalar unknowns
(`g_Psi`, `g_Omega`), solved by `solve_equations_finite_centroids`.
For this reason the centroids are not an input of the script.

The main driver computes the Hermite coefficients → packs them into `params` →
calls the solver → writes the density, the spectral edges, and the training loss.

---

## Activation function

The activation of the random features is selected with `-activation`:

- **`tanh`** (default) — odd, so the first Hermite coefficient `alpha` is zero.
- **`relu`** — `alpha ≠ 0`. It enters the saddle-point equations through the
  centred second moment `beta - alpha²`.

The activation only changes the Hermite coefficients. These are evaluated
numerically for any activation, so the rest of the code is unchanged.

**Exact ReLU coefficients.** For ReLU the coefficients have closed forms
(implemented in `relu_exact_coefficients`). To use them instead of numerical
integration, set the in-code flag near the top of the script:

```python
use_exact_relu_coefficients = True
```

**Custom activations.** To use any other activation, add it to the
`ACTIVATIONS` dictionary at the top of the script. It then becomes available
as a `-activation` choice, with its coefficients computed numerically:

```python
ACTIVATIONS = {
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x: np.maximum(x, 0.0),
    'my_activation': lambda x: ...,
}
```

---

## What the script does

1. **Computes the Hermite coefficients** (`alpha`, `beta_tilde`, `gamma`, `beta`)
   that describe how the activation acts on Gaussian inputs, by trapezoidal
   integration (a deterministic, adaptively-refined 2D quadrature for `beta`),
   or with the exact expressions for ReLU.
2. **Solves the self-consistent equations** for the resolvent `g(z)` along a grid
   of eigenvalues `lambda`, tracking the solution from a large imaginary part of
   `z` down to the real axis (homotopy continuation: each solve seeds the next).
   The spectral density is `rho(lambda) = |Im g(lambda)| / pi`.
3. **Locates the spectral edges** `lambda_star`.
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

Run with defaults:

```bash
python saddle_point_spectrum.py
```

Custom run:

```bash
python saddle_point_spectrum.py -d 200 -chi_p 60 -chi_s 30 -weight 0.3 -t 0.1 -activation relu
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
| `-var0` | `0.5` | Variance of cluster 1. |
| `-var1` | `0.25` | Variance of cluster 2. |
| `-t` | `0.01` | Diffusion time `t` (Ornstein–Uhlenbeck: `a_t = e^{-t}`, `Δ_t = 1 - a_t²`). |
| `-npoints_integration` | `5000` | Number of points in the numerical-integration grid. |
| `-activation` | `tanh` | Activation function: `tanh` or `relu` (case-insensitive), or any name added to `ACTIVATIONS`. |

A few parameters are fixed near the top of the script (edit there if needed):
the available activations (`ACTIVATIONS`), the exact-ReLU switch
(`use_exact_relu_coefficients`), the eigenvalue grid (`n_values`,
`lambda_values`), the continuation protocol (`n_iter`), and the training-loss
time grid (`training_times`, learning rate `lr` to rescale the training time if needed).

---

## Output

A nested folder is created under `results_asymptotic/` whose name encodes the
main parameters, e.g. (defaults):

```
results_asymptotic/solver_saddle_point_equations_finite/P=6000_D=100_S=3000/t=0.01_activation=tanh/
```

Inside you get:

| File | Contents |
|------|----------|
| `rho_lambda_analytical_*.txt` | The spectral density: `lambda  rho(lambda)  rho_Omega(lambda)`. |
| `lambda_star_analytical_*.txt` | Spectral edges: `Re g(lambda_star)  lambda_star`. |
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

- The solver handles a 4-variable system per `lambda`; its cost is dominated by
  `n_values` (default `10000`) times the continuation length (`n_iter`). Reduce
  `n_values` for a quick smoke test.
- The one-off Hermite `beta` integral uses adaptive grid refinement and reports
  its estimated error at startup.
