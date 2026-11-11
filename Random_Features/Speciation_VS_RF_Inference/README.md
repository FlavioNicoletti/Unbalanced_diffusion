# Probing the "inference" of the class centroids by a random-feature model of the diffusion score

This repository probes the **inference transition** of a random-feature model
built on a two-cluster Gaussian mixture — a simple model of *class imbalance* in
diffusion generative models. As the diffusion time is scanned backwards, the
empirical feature covariance `U` develops isolated top eigenvalues whose
eigenvectors begin to **align with a single cluster centroid**: the
random-feature model *infers* the class centroids. That alignment ("inference")
is what the script measures. It is a property of the random-feature model,
distinct from the speciation of the diffusion process, whose time scale sets the
probed window (see [The diffusion-time sweep](#the-diffusion-time-sweep)).

The fully-commented code is in the file
[`centroids_SNR_at_speciation_scale.py`](centroids_SNR_at_speciation_scale.py).

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
   `beta`) that describe how the activation function (`tanh` by default, see
   [Activation function](#activation-function)) acts on Gaussian inputs, per
   cluster, with fast 1D/2D trapezoidal integration (JIT-compiled with `numba`).
   With non-zero centroids the coefficients differ across feature dimensions, so
   each cluster yields a `p × 4` array.
4. **Assembles the empirical feature covariance** `U` (`p × p`, sample-based) by
   summing per-sample contributions in parallel **batches** via `joblib`.
5. **Diagonalizes** `U` and records, per diffusion time, the top three
   eigenvalues and the cosine similarities of the top eigenvectors with the
   projected cluster centroids (the "inference signals").
6. **Appends** these signals to a text file under `results_inference/`.

---

## The diffusion-time sweep

Unlike a fixed-time spectral snapshot, this script scans the diffusion time
from `t_ini = 0.8·log(d)` down to `t_fin = 0.1·log(d)` in `n_times = 20` steps.
The range is set relative to `log(d)`, the `d`-dependent time scale of
speciation of the diffusion process, around which the script probes when the
random-feature model infers the class centroids. The noising process is the
Ornstein–Uhlenbeck one: `x_t = a_t·x + sqrt(Delta_t)·η`, with `a_t = e^{-t}` and
`Delta_t = 1 - e^{-2t}`.

---

## Activation function

The activation is selected with `-act` / `--activation` among the keys of the
`ACTIVATIONS` dictionary at the top of the script. By default it contains
`tanh` (the default) and `relu`. The activation enters only through the Hermite
coefficients, which are evaluated numerically, so any other activation can be
used by adding an entry to the dictionary:

```python
ACTIVATIONS = {
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(x, 0.0),
    "my_activation": lambda x: ...,
}
```

The function must accept both scalars and numpy arrays and be compilable by
`numba` (plain numpy expressions such as `np.tanh`, `np.maximum`, `np.exp` are
fine), because the script JIT-compiles it and calls it inside the integration
kernels.

---

## Installation

**Option A — pip** (into your current/virtual environment):

```bash
pip install -r requirements.txt
```

**Option B — conda** (creates an isolated environment):

```bash
conda env create -f environment.yml
conda activate class-imbalance-diffusion-inference
```

Dependencies: `numpy`, `numba`, `joblib`.

---

## Usage

Run with defaults:

```bash
python centroids_SNR_at_speciation_scale.py
```

Custom run (`d=100`, `p=800` features, `400` samples, `10` realizations, batch
size `50`):

```bash
python centroids_SNR_at_speciation_scale.py -d 100 -p 800 -s 400 -r 10 -b 50
```

Same run with ReLU activation:

```bash
python centroids_SNR_at_speciation_scale.py -d 100 -p 800 -s 400 -r 10 -b 50 -act relu
```

Strong under-sampling of the minority class (majority weight `= 1 - d^{-0.5}`):

```bash
python centroids_SNR_at_speciation_scale.py -u_samp_exp 0.5
```

See all options:

```bash
python centroids_SNR_at_speciation_scale.py --help
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
| `-act` | `--activation` | `tanh` | Activation function: a key of `ACTIVATIONS` (`tanh`, `relu` by default; case-insensitive). |

A few parameters are fixed near the top of the script (edit there if needed):
the diffusion-time window (`t_ini`, `t_fin`, `n_times`) and the
integration-grid resolution (`n_grid`, `xi_max`).

---

## Output

A nested folder is created under `results_inference/` whose name encodes the diffusion
type, the centroid factors, the activation, and the sizes, e.g.:

```
results_inference/output_spectral_quantities_sampling_bias_probe_inference_type=OU_centroid1=3.0_centroid2=3.0_activation=tanh/
    P=800_D=100_S=400/
        signals_to_probe_inference_weight=0.5_var1=0.5_var2=0.5_ortho=False_batch.txt
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
  `numba` and parallelized across features. Since the activation is passed to
  them as an argument, they are compiled at every run (about one second) and not
  cached on disk. Cost grows with `n_grid²`
  (the 2D `beta` integral), `p`, the `n_times` swept times, and `n_realizations`.
- The sample loop is parallelized over batches; tune `-b` (batch size) for your
  machine. `joblib` uses all available cores by default.
- For a quick smoke test, reduce `-d`, `-p`, and `-s`.
