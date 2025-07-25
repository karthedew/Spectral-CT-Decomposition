# CU Boulder CSCA 5642 Deep Learning Final Project

This repository contains code and assets for performing tissue decomposition in dual-energy spectral computed tomography (CT) 
using deep learning. The project explores physics-informed convolutional neural networks (CNNs) to accurately segment and classify 
adipose, fibroglandular, and calcified tissues from simulated spectral CT data.

## Overview

This project was developed as part of the [AAPM DL-Spectral CT Challenge](https://www.aapm.org/GrandChallenge/DL-spectral-CT/). The 
goal is to reconstruct three tissue types from ideal, noise-free dual-energy sinogram data using deep learning models that respect 
the underlying physics of X-ray attenuation.

## Dataset

The [project dataset](https://zenodo.org/records/14262737) is a set of Spectral CT breast tissue transmission and image data.

## Project Structure

Spectral-CT-Decomposition/
├── data/ # Raw and processed input data (.npy.gz)
├── fig/ # EDA and result visualizations
├── models/ # UNet256 and UNet512 model definitions
├── src/ # Training, evaluation, and loss functions
├── main.py # CLI for training, evaluation, and EDA
├── train_config.yaml # Optional config for hyperparameter tuning
└── README.md # This file

## Methodology

- **Data Preparation**: Converts dual-energy transmission sinograms into attenuation maps using the Beer–Lambert law and reconstructs images using filtered backprojection.
- **EDA**: Visualizations of tissue attenuation properties and class distributions.
- **Modeling**: Implements two U-Net architectures:
  - `UNet256`: 2 downsampling levels with 128×128 bottleneck
  - `UNet512`: 3 downsampling levels with 64×64 bottleneck
- **Loss Function**: Combines binary cross-entropy with physical tissue composition loss to encourage meaningful segmentation.

## Results

- **Hyperparameter Tuning**: Grid search across learning rate, stride, filter size, etc.
- **Best Validation Loss**:  
  - `UNet256`: 0.0519  
  - `UNet512`: 0.0380

- **Final MAE (UNet512)**:  
  - Adipose: 0.00260  
  - Fibroglandular: 0.00278  
  - Calcification: 0.00003

- **Tissue Composition Accuracy**: Model closely matches ground truth across all tissues.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Spectral-CT-Decomposition.git
cd Spectral-CT-Decomposition
```

### 2. Install Dependencies
```bash
pip install uv
uv sync
```

### 3. Generate the EDA
```bash
uv run python main.py eda --nsamples 100
```

### 4. Run Hyper-parameter tuning
```bash
uv run python3 main.py tune --subsample 0.1 --model UNET256
uv run python3 main.py tune --subsample 0.1 --model UNET512
```
### 5. Train Optimized Model and Parameters
```bash
uv run python3 main.py --verbose train --model UNET512 --epochs 5 --filter-size 4 --lr 0.001 --stride 3 --padding 2
```