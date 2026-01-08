# Training Random Features Diffusion Model - Tutorial

## Overview

This tutorial explains how to use the `Training_random_features_mixture_unbalance.py` script to train a random feature diffusion model on imbalanced multimodal datasets.

## What the Script Does

The script trains a diffusion model using random features to learn the score function of a Gaussian mixture distribution with class imbalance. It implements:

1. **Random Feature Architecture**: Uses a two-layer network with random features (matrix W) and learnable weights (matrix A)
2. **Gradient Descent Training**: Optimizes the learnable matrix A using either SGD or Adam
3. **Performance Tracking**: Monitors three types of losses during training:
   - Train loss
   - Test loss  
   - Error on the score function

## Quick Start

### Basic Usage

Run the script with default parameters:

```bash
python Training_random_features_mixture_unbalance.py
```

This will:
- Train a model with default architecture (ψ_p=2.0, ψ_n=1.0, d=2000)
- Use Ornstein-Uhlenbeck diffusion at time t=0.01
- Train for 5 million epochs with gradient descent
- Save results to `results_training/well_separated_centroids/`

### Custom Parameters

#### Architecture and Dataset

```bash
python Training_random_features_mixture_unbalance.py \
    --psi_p 2.0 \          # ψ_p = p/d (number of random features)
    --psi_n 1.0 \          # ψ_n = n/d (number of training samples)
    --d 2000               # data dimension
```

#### Diffusion Parameters

```bash
python Training_random_features_mixture_unbalance.py \
    --t 0.01 \                    # diffusion time
    --diffusion_type OU           # 'OU' (Ornstein-Uhlenbeck) or 'FD' (Free Diffusion)
```

#### Data Distribution

```bash
python Training_random_features_mixture_unbalance.py \
    --weight 0.5 \      # class weight for positive class (controls imbalance)
    --var1 0.5 \        # variance of first cluster
    --var2 0.5 \        # variance of second cluster
    --r 0.0             # cosine similarity between cluster centers
```

#### Training Options

```bash
python Training_random_features_mixture_unbalance.py \
    --optimizer Adam \          # 'GD' (gradient descent) or 'Adam'
    --num_epochs 5000000 \      # number of training iterations
    --n_realizations 3          # number of independent training runs
```

## Understanding the Results

### Output Directory Structure

Results are saved to: `results_training/{cluster_type}/{run_name}/`

Where:
- `cluster_type`: Either `well_separated_centroids` or `partially_overlapped_centroids`
- `run_name`: Format is `Psi_p={psi_p}_Psi_n={psi_n}_d={d}_t={t}_{optimizer}`

### Output Files

For each configuration, the script saves 6 NumPy arrays (`.npy` files):

1. **Train_loss_weight={w}_var1={v1}_var2={v2}.npy**: Training loss over time
2. **Test_loss_weight={w}_var1={v1}_var2={v2}.npy**: Test loss over time
3. **Error_score_weight={w}_var1={v1}_var2={v2}.npy**: Overall error on score function
4. **Error_score_1_weight={w}_var1={v1}_var2={v2}.npy**: Error on cluster 1
5. **Error_score_2_weight={w}_var1={v1}_var2={v2}.npy**: Error on cluster 2
6. **Training_time_weight={w}_var1={v1}_var2={v2}.npy**: Training time points

Each file has shape `(n_realizations, n_samples)` where samples are taken logarithmically.

### Loading and Analyzing Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: Load results from a specific run
# Path format: results_training/{cluster_type}/{run_name}/{filename}
run_path = 'results_training/well_separated_centroids/Psi_p=2.0_Psi_n=1.0_d=2e+03_t=1e-02_GD'

train_loss = np.load(f'{run_path}/Train_loss_weight=0.5_var1=0.5_var2=0.5.npy')
training_time = np.load(f'{run_path}/Training_time_weight=0.5_var1=0.5_var2=0.5.npy')

# Plot mean across realizations
plt.loglog(training_time.mean(axis=0), train_loss.mean(axis=0))
plt.xlabel('Training Time')
plt.ylabel('Train Loss')
plt.show()
```

## Example Workflows

### 1. Study Class Imbalance Effect

```bash
# Balanced dataset
python Training_random_features_mixture_unbalance.py --weight 0.5

# Imbalanced dataset
python Training_random_features_mixture_unbalance.py --weight 0.9

# Heavily imbalanced
python Training_random_features_mixture_unbalance.py --weight 0.99
```

### 2. Compare Optimizers

```bash
# Gradient Descent
python Training_random_features_mixture_unbalance.py --optimizer GD

# Adam
python Training_random_features_mixture_unbalance.py --optimizer Adam
```

### 3. Multiple Realizations for Statistics

```bash
# Run 10 independent experiments
python Training_random_features_mixture_unbalance.py --n_realizations 10
```

## Key Concepts

### Random Features

The model uses a random feature representation:
- **W**: Random matrix (p × d), fixed during training
- **A**: Learnable matrix (d × p), optimized via gradient descent
- **Activation**: tanh function
- **Score function**: S_A(x) = A @ tanh(W @ x / √d)

### Diffusion Models

Two diffusion types are supported:
- **Free Diffusion (FD)**: Standard Brownian motion
- **Ornstein-Uhlenbeck (OU)**: Mean-reverting process (default)

### Loss Function

The training objective is to minimize:
```
E_η[(√δ_t * S_A(a_t*x + √δ_t*η) + η)²]
```

Where η is Gaussian noise, a_t and δ_t depend on the diffusion type.

## Computational Notes

- Training time scales with O(d × p × n × num_epochs)
- GPU acceleration is automatic if available
- Learning rate is scaled as: `lr * d / δ_t`, where lr=2e-4 and lr=5e-4 are used respectively for POC (Partially overlapped centroids) and WSC (Well separated centroids).
- Logging uses logarithmic sampling (200 points) to reduce overhead

## Troubleshooting

### Out of Memory
- Reduce `d` (dimension)
- Reduce `psi_p` (random features)
- Reduce `psi_n` (training samples)

### Slow Training
- Reduce `num_epochs`
- Use GPU if available
- Increase batch size in evaluation functions

### Numerical Instability
- Check diffusion time `t` (very small values may cause issues)
- Ensure variances are positive
- Use logsumexp for numerical stability (already implemented)

## References

For more details on the theoretical background, see the paper:
"Study of the generalization performances of diffusion models trained on biased multimodal datasets"
