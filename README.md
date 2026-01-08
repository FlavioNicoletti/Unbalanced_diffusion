# Unbalanced_diffusion
Study of the generalization performances of diffusion models trained on biased multimodal datasets

## Getting Started

This repository contains code for training diffusion models trained on imbalanced multimodal datasets.
The repository contains the following folders:

1) Random_Features: study of score-based diffusion through a random feature model.
2) XXX(Chenxiao)
3) XXX(Enrico)

### Quick Start

1)RF:
```bash
cd Random_Features
python Training_random_features_mixture_unbalance.py
```

For detailed usage instructions, see the [Tutorial](Random_Features/TUTORIAL.md).

2)XXX(Chenxiao)

3)XXX(Enrico)

## Repository Structure

- `Random_Features/` - Main code directory
  - `Training_random_features_mixture_unbalance.py` - Training script
  - `TUTORIAL.md` - Comprehensive tutorial on how to use the training script
  - `results_training/` - Output directory for training results

## Features

- Random feature diffusion model implementation
- Support for imbalanced class distributions
- Multiple optimizer options (SGD, Adam)
- Two diffusion types (Free Diffusion, Ornstein-Uhlenbeck)
- Comprehensive logging and evaluation metrics
- Multiple independent training realizations
