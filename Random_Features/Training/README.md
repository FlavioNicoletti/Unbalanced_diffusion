# Training a random-feature score model on an imbalanced Gaussian mixture

This repository trains a **random-feature score model** by (denoising) score
matching on a two-cluster Gaussian mixture — a simple model of *class imbalance*
in the context of diffusion generative models — and records its **train / test /
score-matching loss curves** as functions of training time.

The fully-commented code is in the file
[`Training_random_features_mixture_unbalance.py`](Training_random_features_mixture_unbalance.py).

---

## The model

The score network is a one-hidden-layer random-feature map

```
S_A(x) = A · phi( W x / sqrt(d) )
```

where the random features `W` (shape `p × d`) are **frozen** at initialization
and only the readout matrix `A` (shape `d × p`) is **trained**.

The activation `phi` is selected with `-activation` among the entries of the
`ACTIVATIONS` dictionary at the top of the script, which by default contains
`tanh` (the default) and `relu`. To use another activation, add an entry, e.g.

```python
ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
}
```

Any function acting elementwise on torch tensors works.

Data is corrupted with a diffusion process `x_t = a_t·x + sqrt(delta_t)·η`
(`η ~ N(0, I)`). Two noising schedules are supported:

- **OU** (Ornstein–Uhlenbeck): `a_t = exp(-t)`, `delta_t = 1 - a_t²`
- **FD** (Free Diffusion): `a_t = 1`, `delta_t = t`

For a Gaussian mixture the **exact (Bayes-optimal) score** of the noised density
is known in closed form; the script uses it to measure how close the trained
model gets to the optimum.

---

## What the script does

For each of several random *realizations* it:

1. **Draws a dataset** of `n = χ_s · d` points in dimension `d`, split into two
   Gaussian clusters with tunable mixing weight, variances, and centroid
   geometry (their cosine similarity).
2. **Builds a frozen random projection** `W` of shape `p × d` (with `p = χ_p · d`)
   and a learnable readout `A` initialized at zero.
3. **Trains `A`** by gradient descent (plain SGD or Adam) on the denoising
   score-matching objective, drawing fresh noise every iteration.
4. At **log-spaced checkpoints**, evaluates and records three families of
   metrics:
   - **train loss** — score-matching loss on the training set,
   - **test loss** — same loss on fresh samples (total and per cluster),
   - **score error** — squared distance between the learned score and the exact
     mixture score (total and per cluster).
5. **Appends** all recorded curves to `.npy` files under `results_training/`.

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

Dependencies: `numpy`, `torch`. A CUDA GPU is used automatically if available,
otherwise the script runs on CPU. For a CUDA-enabled PyTorch build, follow the
[official instructions](https://pytorch.org/get-started/locally/).

---

## Usage

Run with defaults:

```bash
python Training_random_features_mixture_unbalance_userfriendly.py
```

Custom run (`d=1000`, `χ_p=2`, `3e6` iterations, Adam, `5` realizations):

```bash
python Training_random_features_mixture_unbalance_userfriendly.py \
    -d 1000 -chi_p 2 -num_epochs 3000000 -optimizer Adam -n_realizations 5
```

Same run with ReLU activation:

```bash
python Training_random_features_mixture_unbalance_userfriendly.py \
    -d 1000 -chi_p 2 -num_epochs 3000000 -optimizer Adam -n_realizations 5 -activation relu
```

See all options:

```bash
python Training_random_features_mixture_unbalance_userfriendly.py --help
```

---

## Command-line options

| Flag | Long form | Default | Meaning |
|------|-----------|---------|---------|
| `-chi_p` | `--chi_p` | `2.0` | `χ_p = p/d`: number of random features per dimension. |
| `-chi_s` | `--chi_s` | `1.0` | `χ_s = n/d`: number of training samples per dimension. |
| `-d` | `--d` | `2000` | Ambient data dimension `d`. |
| `-t` | `--t` | `0.01` | Diffusion time `t`. |
| `-diffusion_type` | `--diffusion_type` | `OU` | Noising schedule: `OU` (Ornstein–Uhlenbeck) or `FD` (Free Diffusion). |
| `-weight` | `--weight` | `0.5` | Mixing weight of cluster 1 (cluster 2 gets `1 - weight`); the imbalance knob. |
| `-var1` | `--var1` | `0.5` | Variance of cluster 1. |
| `-var2` | `--var2` | `0.5` | Variance of cluster 2. |
| `-r` | `--r` | `0.0` | Cosine similarity between the two cluster centroids, in `[-1, 1]`. |
| `-centroids_factor1` | `--centroids_factor1` | `1.0` | Length scale of centroid 1: 'centroid factor' is the norm of centroid1 divided by sqrt(d). |
| `-centroids_factor2` | `--centroids_factor2` | `1.0` | Length scale of centroid 2. 'centroid factor' is the norm of centroid2 divided by sqrt(d). |
| `-activation` | `--activation` | `tanh` | Activation function of the random features: a key of `ACTIVATIONS` (`tanh`, `relu` by default; case-insensitive). |
| `-optimizer` | `--optimizer` | `GD` | Optimizer: `GD` (plain SGD) or `Adam`. |
| `-num_epochs` | `--num_epochs` | `5000000` | Number of gradient-descent iterations. |
| `-output_dir` | `--output_dir` | `results_training/results_training_RF/` | Directory to save results. |
| `-n_realizations` | `--n_realizations` | `1` | Number of independent training realizations. |

A few quantities are set inside the script (edit there if needed): the top-level
`main_dir` output root, the learning-rate scaling
`lr = 5e-5 · d / delta_t` (chosen because the gradient is very small at
initialization, where `A = 0`), the number of log-spaced checkpoints (`600`),
and the evaluation batch size (`100`).

---

## Output

A nested folder is created under `results_training/results_training_RF/` whose
name encodes the centroid factors and the run parameters (including the
activation, so runs with different activations never share files), e.g.:

```
results_training/results_training_RF/
    centroids_factor1_1.0_centroids_factor2_1.0/
        Chi_p=2.0_Chi_s=1.0_d=2e+03_t=1e-02_GD_activation=tanh/
```

Inside, per parameter set, you get eight `.npy` files (the `weight`, `var1`,
`var2` values are encoded in each filename):

| File | Contents |
|------|----------|
| `Train_loss_*.npy` | Train (score-matching) loss vs checkpoint. |
| `Test_loss_*.npy` | Test loss on the full mixture. |
| `Test_loss_1_*.npy` | Test loss restricted to cluster 1. |
| `Test_loss_2_*.npy` | Test loss restricted to cluster 2. |
| `Error_score_*.npy` | Error vs the exact mixture score (total). |
| `Error_score_1_*.npy` | Score error restricted to cluster 1. |
| `Error_score_2_*.npy` | Score error restricted to cluster 2. |
| `Training_time_*.npy` | Rescaled training-time axis (`epoch × learning_rate`). |

Each array is 2D with **one row per realization**.

> **Note:** output files are opened in *append* mode, so re-running with the
> same parameters adds new rows rather than overwriting. Delete or move the
> folder for a clean re-run.

---

## Reproducibility

By default the script seeds PyTorch from OS entropy, so each run differs (the
seed is printed at startup). To make a run fully reproducible, replace the
seeding line in `main`:

```python
seed = secrets.randbits(64)
```

with a fixed integer, e.g. `seed = 12345`. A re-seed also happens between
realizations so each draws independent randomness; fix that line too for a
deterministic multi-realization run.

---

## Performance notes

- The cost scales with `num_epochs` (each step is one forward/backward pass on
  the `n × d` training set) and with the `600` evaluation checkpoints.
- Test-loss and score-error evaluation draw large fresh sample sets
  (`100·n` and `10 000` points) processed in mini-batches of `100`.
- A CUDA GPU is used automatically when available and gives a large speed-up at
  the default `d = 2000`; on CPU, reduce `-d` and `-num_epochs` for quick tests.
