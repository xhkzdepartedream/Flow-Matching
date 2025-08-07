# Flow-Matching: 高效的连续时间扩散模型

[English](README.md) | [中文](README_zh.md)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个轻量级且高效的 **Flow-Matching** PyTorch实现框架。Flow-Matching是一种先进的连续时间扩散模型方法。本项目旨在展示如何用更少的模型参数和计算资源实现高质量的图像生成。

<p align="center">
  <img src="visuals_celebahq/myplot12.png" width="200"/>
  <img src="visuals_celebahq/plot_2025-08-07 13-30-26_7.png" width="200"/> 
</p>

下载我们训练好的模型：
[点击下载模型权重](https://drive.google.com/file/d/1jzkRGL_ZqgXaFskdXpLxoU__KSgnMz4X/view?usp=drive_link)

## 🏆 性能亮点

本项目展示了卓越的效率，在高清数据集上以极小的模型和极短的训练时间取得了优异的FID分数。

-   **数据集**: CelebA-HQ 256x256
-   **FID 分数**: **47.39**
-   **模型**: DiT (Diffusion Transformer)，仅 **38.75M** 参数
-   **训练时间**: 在6块NVIDIA RTX 4090上仅需 **约20小时**

这些结果有力地证明了，在潜空间中将Flow-Matching与高效的DiT架构相结合的强大能力，使得高质量生成模型的门槛大大降低。

## 🚀 核心特性

-   **高效且快速**: 在CelebA-HQ 256x256数据集上，仅需约20小时训练即可达到47.39的FID分数。
-   **轻量级模型**: 使用的DiT模型仅有 **38.75M** 参数，远小于执行同类任务的典型模型。
-   **前沿架构**: 实现了在VAE潜空间中，结合Flow-Matching与Diffusion Transformer (DiT) 的先进框架。
-   **模块化与配置驱动**: 通过YAML配置文件，可以轻松地进行不同模型和调度器的实验。
-   **多GPU支持**: 包含了为分布式训练和采样优化的脚本。

## 🔧 模型配置与训练

本项目的成功关键在于其高效的模型配置。我们采用两阶段方法：

1.  **VAE压缩**: 使用一个精调的 `AutoencoderKL` (VAE) 将CelebA-HQ数据集中的256x256x3高分辨率图像压缩成一个更小的32x32x4潜空间表示。这极大地降低了扩散模型的计算负担。

2.  **DiT (Diffusion Transformer)**: 在潜空间上训练一个具有以下配置的 `DiT` 模型：
    -   **输入尺寸**: 32x32
    -   **输入通道数**: 4
    -   **Patch尺寸**: 2
    -   **模型维度**: 512
    -   **Transformer模块数**: 8
    -   **注意力头数**: 16
    -   **总参数量**: **38.75M**

该配置使我们能够在最小化计算和内存需求的同时，实现高质量的图像生成，是高效扩散模型设计的绝佳范例。

## 📁 项目结构

```
Flow-Matching/
├── configs/                  # YAML 配置文件
├── data_processing/          # 数据加载与预处理脚本
├── fm_scheduler/             # Flow-Matching 调度器与采样器
├── models/                   # 神经网络架构 (DiT, U-Net)
├── modules/                  # 可复用的神经网络组件
├── scripts/                  # 执行脚本 (训练, 采样)
├── trainer/                  # 训练流程控制类
├── utils.py                  # 工具函数
├── requirements.txt
└── README.md
```

## 🛠️ 安装

### 环境要求
- Python 3.8 或更高版本
- 支持CUDA的GPU (推荐)

### 快速设置

1.  **克隆仓库**
```bash
git clone https://github.com/your-username/Flow-Matching.git
cd Flow-Matching
```
并从上面的链接下载训练好的模型。

2.  **安装依赖**
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 选择配置

从 `configs/` 目录中选择一个配置。例如，`celebahq_dit.yaml` 是为CelebA-HQ数据集设计的。

### 2. 生成样本

使用 `scripts/batch_sample.py` 脚本生成图像。你可以在单GPU或多GPU模式下运行。

**单GPU:**
```bash
python scripts/batch_sample.py
```

**多GPU (例如, 使用2个GPU):**
```bash
torchrun --nproc_per_node=2 scripts/batch_sample.py
```

## 📚 引用

如果您在研究中使用了本项目的代码，请引用原始论文和本代码库。

## 📄 许可证

本项目采用 MIT 许可证。
