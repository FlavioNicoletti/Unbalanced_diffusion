# Spectral analysis of a diffusion feature matrix for a Gaussian mixture

This repository computes and stores the **spectral properties** and
**semi-analytical learning curves** of a random-feature model built on a
two-cluster Gaussian mixture — a simple model of *class imbalance* in the
context of diffusion generative models.

The fully-commented code is in the file
[`GEP_gaussian_mixture_UDiff_batch_userfriendly.py`](GEP_gaussian_mixture_UDiff_batch_userfriendly.py).

---

## What the script does

For each of several random *realizations* it:

1. **Draws a dataset** of `n_samples` points in dimension `d`, split into two
   Gaussian clusters with tunable mixing weight, variances, and centroid
   geometry (their cosine similarity).
2. **Builds a random projection matrix** `W` of shape `p × d` (the "random
   features").
3. **Computes Hermite coefficients** (`alpha`, `beta`, `beta_tilde`,
   `gamma_tilde`) that describe how the activation function (`tanh` by default,
   see [Activation function](#activation-function)) acts on Gaussian
   inputs. These are obtained for each cluster with fast 1D/2D trapezoidal
   numerical integration (JIT-compiled with `numba`). For non-zero centroids,
   the coefficients are different for each hidden feature dimension, therefore
   the code computes 4 `p`-dimensional vectors per cluster. For zero centroids, the coefficients
   are scalar, and the code computes 4 scalar quantities per cluster.
4. **Assembles two `p × p` matrices**:
   - `U`        — the empirical (sample-based) feature covariance,
   - `U_tilde`  — its population / generalization counterpart (sample-free).
5. **Diagonalizes** both matrices and derives **training / test / score-matching
   loss curves** as functions of training time.
6. **Writes** all spectral quantities and loss curves to text files under
   `results_spectrum/`.

The expensive per-sample loop is split into **batches** that run in parallel via
`joblib`.

---

## Activation function

The activation is selected with `-act` / `--activation` among the keys of the
`ACTIVATIONS` dictionary at the top of the script. By default it contains:

- **`tanh`** (default) — odd, so `alpha = 0` for zero centroids;
- **`relu`** — `alpha ≠ 0`.

The activation enters only through the Hermite coefficients, which are always
evaluated numerically; the construction of `U` and `U_tilde` is the same for
every activation. To use another activation, add an entry to the dictionary:

```python
ACTIVATIONS = {
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(x, 0.0),
    "my_activation": lambda x: ...,
}
```

and run with `-act my_activation`. The function must accept both scalars and
numpy arrays and be compilable by `numba` (plain numpy expressions such as
`np.tanh`, `np.maximum`, `np.exp` are fine), because the script JIT-compiles it
and calls it inside the integration kernels.

---

## Installation

**Option A — pip** (into your current/virtual environment):

```bash
pip install -r requirements.txt
```

**Option B — conda** (creates an isolated environment):

```bash
conda env create -f environment.yml
conda activate class-imbalance-diffusion
```

Dependencies: `numpy`, `numba`, `joblib`, `matplotlib` (the last is only needed
if you re-enable the optional plotting block at the end of the script).

---

## Usage

Run with defaults:

```bash
python GEP_gaussian_mixture_UDiff_batch_userfriendly.py
```

Custom run (`d=100`, `p=800` features, `400` samples, `10` realizations, batch
size `50`):

```bash
python GEP_gaussian_mixture_UDiff_batch_userfriendly.py -d 100 -p 800 -s 400 -r 10 -b 50
```

Same run with ReLU activation:

```bash
python GEP_gaussian_mixture_UDiff_batch_userfriendly.py -d 100 -p 800 -s 400 -r 10 -b 50 -act relu
```

See all options:

```bash
python GEP_gaussian_mixture_UDiff_batch_userfriendly.py --help
```

---

## Command-line options

| Flag | Long form | Default | Meaning |
|------|-----------|---------|---------|
| `-d`  | `--dimension`        | `100`  | Ambient dimension `d` of each data point. |
| `-p`  | `--projections`      | `800`  | Number `p` of random features / projections. |
| `-s`  | `--samples`          | `400`  | Total samples drawn per realization. |
| `-r`  | `--realizations`     | `10`   | Number of independent random realizations. |
|       | `--ortho`            | off    | Use orthogonal centroids (cosine = 0). Ignored if `-cc` is given. When this is set to false, centroids are anti-aligned. |
| `-cc` | `--centroid_cosine`  | `None` | Cosine similarity between the two centroids, in `[-1, 1]`. Overrides `--ortho`. |
| `-b`  | `--batch_size`       | `50`   | Samples processed together in one parallel batch. |
| `-j`  | `--jobs`             | `-1`   | Parallel worker processes (`-1` = all CPU cores). |
| `-w`  | `--weight`           | `0.5`  | Mixing weight of cluster 1 (cluster 2 gets `1 - weight`). |
| `-var1` | `--var1`           | `0.5`  | Variance of cluster 1. |
| `-var2` | `--var2`           | `0.5`  | Variance of cluster 2. |
| `-c1` | `--centroid_factor1` | `3.0`  | Length scale of centroid 1. |
| `-c2` | `--centroid_factor2` | `3.0`  | Length scale of centroid 2. |
| `-act` | `--activation`      | `tanh` | Activation function: a key of `ACTIVATIONS` (`tanh`, `relu` by default; case-insensitive). |

A few parameters are fixed near the top of the script (edit there if needed):
the diffusion time `t_diffusion`, the diffusion type (`"OU"` or `"FD"`), the
learning rate `lr` (used to reparemeterize the training time in learning curves), the centroid construction mode (`homogeneous_centroids`),
and the integration-grid resolution (`n_grid`, `xi_max`).

---

## Output

A nested folder is created under `results_spectrum/` whose name encodes the main
parameters (including the activation), e.g.:

```
results_spectrum/output_spectral_quantities_sampling_bias_t_diff=0.001_type=OU_centroid1=3.0_centroid2=3.0_activation=tanh/P=800_D=100_S=400/
```

(when centroids are non-orthogonal, a further `cosine_similarity=.../`
sub-folder is added). Inside, per parameter set, you get:

| File | Contents |
|------|----------|
| `spectral_quantities_*.txt`            | per-eigenvalue: index, eigenvalue(`U`), IPR(`U`), eigenvalue(`U_tilde`), IPR(`U_tilde`), `nu` diagonal |
| `spectral_quantities_cluster1/2_*.txt` | same, restricted to each cluster |
| `hermite_integrals_cluster1/2_*.txt`   | the precomputed coefficients `[alpha, beta_tilde, gamma_tilde, beta]` |
| `analytical_loss_curves_*.txt`         | per training time: train / test / score losses (total + per cluster) |
| `omegas_*.txt`                         | overlap quantities `‖ω₁‖²`, `‖ω₂‖²`, `ω₁·ω₂` per eigenvector |

> **Note:** output files are opened in *append* mode, so re-running with the
> same parameters adds new rows rather than overwriting. Delete or move the
> folder for a clean re-run.

---

## Reproducibility

By default the script seeds its random number generator from OS entropy, so
each run differs (the seed used is printed at startup). To make a run fully
reproducible, replace the seeding line near the top:

```python
seed = secrets.randbits(128)
```

with a fixed integer, e.g. `seed = 12345`. As an alternative, past runs can be reproduced
by using the seed printed on their log files.

---

## Performance notes

- The Hermite integrals are the heaviest step; they are JIT-compiled with
  `numba` and parallelized across features. Since the activation is passed to
  them as an argument, they are compiled at every run (about one second); they
  are not cached on disk.
- The sample loop scales with `n_samples` and is parallelized over batches; tune
  `-b` (batch size) and `-j` (jobs) for your machine.
