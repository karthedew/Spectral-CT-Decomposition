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

```bash
.
├── data
│   ├── 14262737.zip
│   ├── attn_images.npy
│   ├── attn_images.npy.gz
│   ├── highkVpImages.npy.gz
│   ├── highkVpTransmission.npy.gz
│   ├── lowkVpImages.npy.gz
│   ├── lowkVpTransmission.npy.gz
│   ├── Phantom_Adipose.npy.gz
│   ├── Phantom_Calcification.npy.gz
│   ├── Phantom_Fibroglandular.npy.gz
│   └── starting_kit.tgz
├── doc
│   ├── abstract
│   │   └── abstract.tex
│   ├── conclusion
│   │   └── conclusion.tex
│   ├── discussion
│   │   └── discussion.tex
│   ├── evaluation
│   │   └── evaluation.tex
│   ├── fig
│   │   ├── attenuation_histograms.png
│   │   ├── attenuation_scatter.png
│   │   ├── calcification_distribution.png
│   │   └── tissue_percentage_distribution.png
│   ├── introduction
│   │   └── introduction.tex
│   ├── methodology
│   │   └── methodology.tex
│   ├── plots
│   │   ├── bitcoin_price_plot.png
│   │   ├── correlation_matrix_plot.png
│   │   ├── fear_and_greed_plot.png
│   │   ├── m2sl_plot.png
│   │   ├── prediction_vs_actual.png
│   │   └── trailing_average_plot.png
│   ├── presentation.odp
│   ├── presentation.pdf
│   ├── references.bib
│   ├── related_work
│   │   └── related_work.tex
│   ├── Spectral_CT_Decomposition.aux
│   ├── Spectral_CT_Decomposition.bbl
│   ├── Spectral_CT_Decomposition.fdb_latexmk
│   ├── Spectral_CT_Decomposition.fls
│   ├── Spectral_CT_Decomposition.log
│   ├── Spectral_CT_Decomposition.out
│   ├── Spectral_CT_Decomposition.pdf
│   ├── Spectral_CT_Decomposition.synctex.gz
│   └── Spectral_CT_Decomposition.tex
├── __init__.py
├── main.py
├── Makefile
├── pyproject.toml
├── README.md
├── src
│   ├── AttnDataset.py
│   ├── CNN.py
│   ├── CosMLP.py
│   ├── Dataset.py
│   ├── EDA.py
│   ├── fanbeam_fbp_sino.py
│   ├── ImageDataset.py
│   ├── __init__.py
│   ├── MLP.py
│   ├── __pycache__
│   │   ├── AttnDataset.cpython-311.pyc
│   │   ├── AttnDataset.cpython-312.pyc
│   │   ├── CNN.cpython-311.pyc
│   │   ├── CNN.cpython-312.pyc
│   │   ├── CosMLP.cpython-311.pyc
│   │   ├── CosMLP.cpython-312.pyc
│   │   ├── Dataset.cpython-311.pyc
│   │   ├── Dataset.cpython-312.pyc
│   │   ├── EDA.cpython-311.pyc
│   │   ├── EDA.cpython-312.pyc
│   │   ├── fanbeam_fbp_sino.cpython-311.pyc
│   │   ├── fanbeam_fbp_sino.cpython-312.pyc
│   │   ├── ImageDataset.cpython-312.pyc
│   │   ├── __init__.cpython-311.pyc
│   │   ├── __init__.cpython-312.pyc
│   │   ├── MLP.cpython-311.pyc
│   │   ├── MLP.cpython-312.pyc
│   │   ├── PINN.cpython-312.pyc
│   │   ├── Trainer.cpython-311.pyc
│   │   └── Trainer.cpython-312.pyc
│   ├── Trainer.py
│   └── tuning.py
└── uv.lock

14 directories, 77 files
```

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