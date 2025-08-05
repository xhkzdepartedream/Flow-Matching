# Flow-Matching: Efficient Continuous-Time Diffusion Models

[English](README.md) | [中文](README_zh.md)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A lightweight and efficient PyTorch framework for **Flow-Matching**, a modern approach to continuous-time diffusion models. This project demonstrates how to achieve high-quality image generation with significantly fewer parameters and computational resources.

<p align="center">
  <img src="visuals_celebahq/myplot7.png" width="200"/>
  <img src="visuals_celebahq/myplot9.png" width="200"/> 
</p>

## 📁 Project Structure

```
Flow-Matching/
├── configs/                  # YAML configuration files
├── data_processing/          # Data loading and preprocessing scripts
├── fm_scheduler/             # Flow-Matching schedulers and samplers
├── models/                   # Neural network architectures (DiT, U-Net)
├── modules/                  # Reusable neural network components
├── scripts/                  # Execution scripts (training, sampling)
├── trainer/                  # Training orchestration classes
├── utils.py                  # Utility functions
├── requirements.txt
└── README.md
```

## 🚀 Key Features

- **Efficient Flow-Matching**: Implements a cutting-edge continuous-time diffusion model that is both fast and resource-friendly.
- **High-Quality Generation**: Achieves excellent results on the CelebA-HQ dataset.
- **Optimized Model Configuration**: A fine-tuned VAE compresses 256x256x3 images into a 32x32x4 latent space, which is then processed by a DiT model with only **38.75M** parameters.
- **Modular and Config-Driven**: Easily experiment with different models and schedulers using YAML configuration files.
- **Multi-GPU Support**: Includes scripts for distributed training and sampling.

## 🔧 Model Configuration & Training

This project's success lies in its efficient model configuration. We use a two-stage approach:

1.  **VAE Compression**: A fine-tuned `AutoencoderKL` (VAE) is used to compress high-resolution 256x256x3 images from the CelebA-HQ dataset into a much smaller 32x32x4 latent representation. This significantly reduces the computational load for the diffusion model.

2.  **Diffusion Transformer (DiT)**: A `DiT` model with the following configuration is trained on the latent space:
    -   **Input Size**: 32x32
    -   **Input Channels**: 4
    -   **Patch Size**: 2
    -   **Model Dimension**: 512
    -   **Transformer Blocks**: 8
    -   **Attention Heads**: 16
    -   **Total Parameters**: **38.75M**

This setup allows us to achieve high-quality image generation while minimizing computational and memory requirements, making it an excellent example of efficient diffusion model design.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Quick Setup

1.  **Clone the repository**
```bash
git clone https://github.com/your-username/Flow-Matching.git
cd Flow-Matching
```

2.  **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Choose Your Configuration

Select a configuration from the `configs/` directory. For example, `celebahq_dit.yaml` is configured for the CelebA-HQ dataset.

### 2. Generate Samples

Use the `batch_sample.py` script to generate images. You can run it in single-GPU or multi-GPU mode.

**Single-GPU:**
```bash
python scripts/batch_sample.py
```

**Multi-GPU (e.g., with 2 GPUs):**
```bash
torchrun --nproc_per_node=2 scripts/batch_sample.py
```

## 📚 Citation

If you use this code in your research, please cite the original papers and this repository.

## 📄 License

This project is licensed under the MIT License.

---

## Thus, the project is completed, and all has been said.