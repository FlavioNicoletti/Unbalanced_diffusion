# Spectral analysis of a diffusion feature matrix for a Gaussian mixture

This repository computes and stores the **spectral properties** and
**semi-analytical learning curves** of a random-feature model built on a
two-cluster Gaussian mixture — a simple model of *class imbalance* in the
context of diffusion generative models.

The fully-commented code is in the file
[`Compute_U_spectrum_and_SemiAnalytical_Learning_Curves.py`](Compute_U_spectrum_and_SemiAnalytical_Learning_Curves.py).

---

## What the script does

For each of several random *realizations* it:

1. **Draws a dataset** of `n_samples` points in dimension `d`, split into two
   Gaussian clusters with tunable mixing weight, variances, and centroid
   geometry (their cosine similarity).
2. **Builds a random projection matrix** `W` of shape `p × d` (the "random
   features").
3. **Computes Hermite coefficients** (`alpha`, `beta`, `beta_tilde`,
   `gamma_tilde`) that describe how a `tanh` non-linearity acts on Gaussian
   inputs. These are obtained for each cluster with fast 1D/2D trapezoidal
   numerical integration (JIT-compiled with `numba`). For non-zero centroids,
   the coefficients are different for each hidden feature dimension, therefore
   the code computes 4 `p`-dimensional vectors per cluster. For zero centroids, the coefficients
   are scalar, and the code computes 4 scalar quantities per cluster.
4. **Assembles two `p × p` matrices**:
   - `U`        — the empirical (sample-based) feature covariance,
   - `U_tilde`  — its population / generalization counterpart (sample-free).
5. **Diagonalizes** both matrices and derives **training / test / score-matching
   loss curves** as functions of training time. The expression of the learning curves 
   can be found in Appendices G 'Theoretical curves of train and
   test losses' and H 'Score MSE' of the paper.
7. **Writes** all spectral quantities and loss curves to text files under
   `results/`.

The expensive per-sample loop is split into **batches** that run in parallel via
`joblib`.

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
python Compute_U_spectrum_and_SemiAnalytical_Learning_Curves.py
```

Custom run (`d=100`, `p=800` features, `400` samples, `10` realizations, batch
size `50`):

```bash
python Compute_U_spectrum_and_SemiAnalytical_Learning_Curves.py -d 100 -p 800 -s 400 -r 10 -b 50
```

See all options:

```bash
python Compute_U_spectrum_and_SemiAnalytical_Learning_Curves.py --help
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

A few parameters are fixed near the top of the script (edit there if needed):
the diffusion time `t_diffusion`, the diffusion type (`"OU"` or `"FD"`), the
learning rate `lr` (used to reparemeterize the training time in learning curves), the centroid construction mode (`homogeneous_centroids`),
and the integration-grid resolution (`n_grid`, `xi_max`).

---

## Output

A nested folder is created under `results/` whose name encodes the main
parameters, e.g.:

```
results/output_spectral_quantities_sampling_bias_t_diff=0.001_type=OU_centroid1=3.0_centroid2=3.0/P=800_D=100_S=400/
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
  `numba` and parallelized across features. The **first** call pays a one-time
  compilation cost (cached on disk via `cache=True`).
- The sample loop scales with `n_samples` and is parallelized over batches; tune
  `-b` (batch size) and `-j` (jobs) for your machine.
