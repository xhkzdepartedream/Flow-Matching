# Flow-Matching: 高效的连续时间扩散模型

[English](README.md) | [中文](README_zh.md)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个轻量级且高效的 **Flow-Matching** PyTorch 实现，这是一种先进的连续时间扩散模型。本项目展示了如何用更少的参数和计算资源实现高质量的图像生成。

<p align="center">
  <img src="visuals_celebahq/myplot7.png" width="200"/>
  <img src="visuals_celebahq/myplot9.png" width="200"/> 
</p>

## 📁 项目结构

```
Flow-Matching/
├── configs/                  # YAML 配置文件
├── data_processing/          # 数据加载与预处理脚本
├── fm_scheduler/             # Flow-Matching 调度器与采样器
├── models/                   # 神经网络架构 (DiT, U-Net)
├── modules/                  # 可复用的神经网络组件
├── scripts/                  # 执行脚本 (训练, 采样)
├── trainer/                  # 训练流程管理类
├── utils.py                  # 工具函数
├── requirements.txt
└── README.md
```

## 🚀 核心特性

- **高效的 Flow-Matching**: 实现了一种前沿的连续时间扩散模型，兼具速度与资源效率。
- **高质量生成**: 在 CelebA-HQ 数据集上取得了出色的生成效果。
- **优化的模型配置**: 使用一个精调的 VAE 将 256x256x3 的图像压缩至 32x32x4 的潜在空间，并由一个仅有 **38.75M** 参数的 DiT 模型进行处理。
- **模块化与配置驱动**: 使用 YAML 配置文件，轻松进行不同模型和调度器的实验。
- **多GPU支持**: 包含了用于分布式训练和采样的脚本。

## 🔧 模型配置与训练

本项目的成功关键在于其高效的模型配置。我们采用两阶段方法：

1.  **VAE 压缩**: 使用一个精调的 `AutoencoderKL` (VAE) 将 CelebA-HQ 数据集中的高分辨率 256x256x3 图像压缩为更小的 32x32x4 潜在表示。这极大地降低了扩散模型的计算负担。

2.  **扩散变换器 (DiT)**: 在潜在空间上训练一个具有以下配置的 `DiT` 模型：
    -   **输入尺寸**: 32x32
    -   **输入通道**: 4
    -   **Patch 尺寸**: 2
    -   **模型维度**: 768
    -   **Transformer 块**: 12
    -   **注意力头**: 16
    -   **总参数量**: **38.75M**

这一设计使我们能够在最小化计算和内存需求的同时，实现高质量的图像生成，是高效扩散模型设计的绝佳范例。

## 🛠️ 安装

### 环境要求
- Python 3.8 或更高版本
- 支持 CUDA 的 GPU (推荐)

### 快速设置

1.  **克隆仓库**
```bash
git clone https://github.com/your-username/Flow-Matching.git
cd Flow-Matching
```

2.  **安装依赖**
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 选择配置

从 `configs/` 目录中选择一个配置。例如，`celebahq_dit.yaml` 是为 CelebA-HQ 数据集配置的。

### 2. 生成样本

使用 `batch_sample.py` 脚本生成图像。你可以在单GPU或多GPU模式下运行。

**单GPU:**
```bash
python scripts/batch_sample.py
```

**多GPU (例如, 使用 2 个GPU):**
```bash
torchrun --nproc_per_node=2 scripts/batch_sample.py
```

## 📚 引用

如果您在研究中使用了此代码，请引用原始论文和本仓库。

## 📄 许可证

本项目采用 MIT 许可证。

---

## 至是，工程已毕，言尽于此。
