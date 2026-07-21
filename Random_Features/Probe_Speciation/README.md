# Probing "speciation" in a diffusion feature matrix for a Gaussian mixture

This repository probes the **speciation transition** of a random-feature model
built on a two-cluster Gaussian mixture — a simple model of *class imbalance* in
diffusion generative models. As the diffusion time is scanned backwards, the
empirical feature covariance `U` develops isolated top eigenvalues whose
eigenvectors begin to **align with a single cluster centroid**. That alignment
("speciation") is what the script measures.

The fully-commented code is in the file
[`probe_speciation.py`](probe_speciation.py).

---

## What the script does

For each of several random *realizations*, and within each a **sweep of
diffusion times**, it:

1. **Draws a dataset** of `n_samples` points in dimension `d`, split into two
   Gaussian clusters with tunable mixing weight, variances, and centroid length
   scales. Strong under-sampling of the minority class can be requested via an
   exponent (majority weight = `1 - d^{-exponent}`).
2. **Builds a random projection matrix** `W` of shape `p × d` (the "random
   features").
3. **Computes Hermite coefficients** (`alpha`, `beta_tilde`, `gamma_tilde`,
   `beta`) that describe how a `tanh` non-linearity acts on Gaussian inputs, per
   cluster, with fast 1D/2D trapezoidal integration (JIT-compiled with `numba`).
   With non-zero centroids the coefficients differ across feature dimensions, so
   each cluster yields a `p × 4` array.
4. **Assembles the empirical feature covariance** `U` (`p × p`, sample-based) by
   summing per-sample contributions in parallel **batches** via `joblib`.
5. **Diagonalizes** `U` and records, per diffusion time, the top three
   eigenvalues and the cosine similarities of the top eigenvectors with the
   projected cluster centroids (the "speciation signals").
6. **Appends** these signals to a text file under `results/`.

---

## The diffusion-time sweep

Unlike a fixed-time spectral snapshot, this script scans the diffusion time
from `t_ini = 0.8·log(d)` down to `t_fin = 0.1·log(d)` in `n_times = 20` steps.
The range is set relative to `log(d)` because the speciation transition sits at
a `d`-dependent time scale. Two noising schedules are available (`diffusion_type`
near the top of the script): `OU` (Ornstein–Uhlenbeck, default) and `FD` (Free
Diffusion).

---

## Installation

**Option A — pip** (into your current/virtual environment):

```bash
pip install -r requirements.txt
```

**Option B — conda** (creates an isolated environment):

```bash
conda env create -f environment.yml
conda activate class-imbalance-diffusion-speciation
```

Dependencies: `numpy`, `numba`, `joblib`.

---

## Usage

Run with defaults:

```bash
python probe_speciation.py
```

Custom run (`d=100`, `p=800` features, `400` samples, `10` realizations, batch
size `50`):

```bash
python probe_speciation.py -d 100 -p 800 -s 400 -r 10 -b 50
```

Strong under-sampling of the minority class (majority weight `= 1 - d^{-0.5}`):

```bash
python probe_speciation.py -u_samp_exp 0.5
```

See all options:

```bash
python probe_speciation.py --help
```

---

## Command-line options

| Flag | Long form | Default | Meaning |
|------|-----------|---------|---------|
| `-d`  | `--dimension`      | `100`  | Ambient dimension `d` of each data point. |
| `-p`  | `--projections`    | `800`  | Number `p` of random features / projections. |
| `-s`  | `--samples`        | `400`  | Total samples drawn per realization. |
| `-r`  | `--realizations`   | `10`   | Number of independent random realizations. |
|       | `--ortho`          | off    | Kept for interface compatibility; here it only affects the output file name. |
| `-b`  | `--batch_size`     | `50`   | Samples processed together in one parallel batch. |
| `-w`  | `--weight`         | `0.5`  | Mixing weight of cluster 1 (cluster 2 gets `1 - weight`). Ignored when `-u_samp_exp > 0`. |
| `-var1` | `--var1`         | `0.5`  | Variance of cluster 1. |
| `-var2` | `--var2`         | `0.5`  | Variance of cluster 2. |
| `-c1` | `--centroid_factor1` | `3.0` | Length scale of centroid 1. |
| `-c2` | `--centroid_factor2` | `3.0` | Length scale of centroid 2. |
| `-u_samp_exp` | `--under_sampling_exponent` | `0.0` | Under-sampling exponent: majority weight `= 1 - d^{-exp}` (0.0 = use fixed `--weight`; must be ≥ 0). |

A few parameters are fixed near the top of the script (edit there if needed):
the diffusion-time window (`t_ini`, `t_fin`, `n_times`), the diffusion type
(`"OU"` or `"FD"`), and the integration-grid resolution (`n_grid`, `xi_max`).

---

## Output

A nested folder is created under `results_speciation/` whose name encodes the diffusion
type, the centroid factors, and the sizes, e.g.:

```
results_speciation/output_spectral_quantities_sampling_bias_probe_speciation_type=OU_centroid1=3.0_centroid2=3.0/
    P=800_D=100_S=400/
        signals_to_probe_speciation_weight=0.5_var1=0.5_var2=0.5_ortho=False_batch.txt
```

(when an under-sampling exponent is used, the file name encodes it instead of
`weight`). Each **row** of the file corresponds to one diffusion time and
contains, in order:

| Column(s) | Contents |
|-----------|----------|
| 1 | diffusion time `t` |
| 2–4 | top three eigenvalues of `U`: `λ₁ ≥ λ₂ ≥ λ₃` |
| 5 | `cos(v₁, μ₁)` — top eigenvector vs projected centroid 1 |
| 6 | `cos(v₁, μ₂)` — top eigenvector vs projected centroid 2 |
| 7 | `cos(v₂, μ₂)` — 2nd eigenvector vs projected centroid 2 |
| 8 | `cos(v₂, μ₁)` — 2nd eigenvector vs projected centroid 1 |
| 9 | `⟨cos(other eigvecs, μ₁)⟩` — average over all remaining eigenvectors |
| 10 | `⟨cos(other eigvecs, μ₂)⟩` — average over all remaining eigenvectors |

Here `v₁ / v₂` are the eigenvectors of the largest / second-largest eigenvalue
of `U`, and `μ₁ / μ₂` are the projected cluster centroids.

> **Note:** the output file is opened in *append* mode, so re-running with the
> same parameters adds new rows rather than overwriting. Delete or move the
> file/folder for a clean re-run.

---

## Reproducibility

By default the script seeds its NumPy random generator from OS entropy, so each
run differs (the seed is printed at startup). To make a run fully reproducible,
replace the seeding line near the top:

```python
seed = secrets.randbits(128)
```

with a fixed integer, e.g. `seed = 12345`. Note that each parallel batch also
draws its own OS-entropy seed inside `process_batch`; fix that too for
bit-for-bit reproducibility.

---

## Performance notes

- The Hermite integrals are the heaviest step; they are JIT-compiled with
  `numba` and parallelized across features. The **first** call pays a one-time
  compilation cost (cached on disk via `cache=True`). Cost grows with `n_grid²`
  (the 2D `beta` integral), `p`, the `n_times` swept times, and `n_realizations`.
- The sample loop is parallelized over batches; tune `-b` (batch size) for your
  machine. `joblib` uses all available cores by default.
- For a quick smoke test, reduce `-d`, `-p`, and `-s`.
