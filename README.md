# Unbalanced_diffusion
Study of the generalization performances of diffusion models trained on biased multimodal datasets

## Getting Started

This repository contains code for training diffusion models trained on imbalanced multimodal datasets.
The repository contains the following folders:

1) Random_Features: study of score-based diffusion through a random feature model. Its subdirectories are:
   - `Training/`: trains the random-feature score model and records train, test, and score-matching loss curves.
   - `Spectrum_Numerical_Semi_Analytical/`: computes numerical spectra and semi-analytical learning curves for the feature covariance matrix.
   - `Spectrum_Asymptotic_Theory/`: computes asymptotic spectra and fully analytical learning curves using self-consistent equations.
   - `Probe_Speciation/`: scans diffusion time to measure eigenvalue isolation and eigenvector alignment with mixture centroids.
2) XXX(Chenxiao)
3) XXX(Enrico)

### Quick Start

1) RF training:
```bash
cd Random_Features/Training
python Training_random_features_mixture_unbalance_userfriendly.py
```

For detailed usage instructions, see the [Training README](Random_Features/Training/README.md).

2)XXX(Chenxiao)

3)XXX(Enrico)

## Repository Structure

- `Random_Features/` - Main random-feature study directory
  - `Training/` - Random-feature score-model training
    - `Training_random_features_mixture_unbalance_userfriendly.py` - Training script
    - `README.md` - Training instructions, options, and output format
    - `results_training/` - Training output directory
  - `Spectrum_Numerical_Semi_Analytical/` - Numerical spectral analysis and semi-analytical learning curves
    - `Compute_U_spectrum_and_SemiAnalytical_Learning_Curves.py` - Spectral-analysis script
    - `README.md` - Usage and output documentation
    - `results_spectrum/` - Spectral-analysis output directory
  - `Spectrum_Asymptotic_Theory/` - Asymptotic spectral analysis and analytical learning curves
    - `saddle_point_spectrum.py` - Self-consistent spectrum solver
    - `README.md` - Usage and output documentation
  - `Probe_Speciation/` - Diffusion-time sweep for speciation signals
    - `probe_speciation.py` - Speciation-probing script
    - `README.md` - Usage and output documentation
    - `results_speciation/` - Speciation output directory

## Features

- Random feature diffusion model implementation
- Support for imbalanced class distributions
- Multiple optimizer options (SGD, Adam)
- Two diffusion types (Free Diffusion, Ornstein-Uhlenbeck)
- Comprehensive logging and evaluation metrics
- Multiple independent training realizations
