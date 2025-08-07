import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import uuid


def save_image_tensor(image_tensor, save_path):
    """
    直接保存图像张量为PNG文件，不进行额外的范围转换
    
    Args:
        image_tensor: 图像张量，形状为[C,H,W]，数值范围应在[0,1]
        save_path: 保存路径
    """
    # 确保张量在CPU上
    image_tensor = image_tensor.detach().cpu()

    # 确保数值范围在[0,1]内
    image_tensor = torch.clamp(image_tensor, 0, 1)

    if image_tensor.shape[0] == 1:
        # 灰度图
        image_np = (image_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        cmap = 'gray'
    elif image_tensor.shape[0] == 3:
        # RGB图
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cmap = None
    else:
        raise ValueError(f"Unsupported image shape: {image_tensor.shape}")

    # 创建 figure，关闭边框和轴线
    fig = plt.figure(figsize = (image_np.shape[1] / 100, image_np.shape[0] / 100), dpi = 100)
    ax = plt.Axes(fig, [0, 0, 1, 1])  # 占满整个画布
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_np, cmap = cmap)

    plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
    plt.close()


def save_cifar10_images(output_dir, num_images = None):
    """
    保存CIFAR-10数据集为图片，用于FID计算
    
    Args:
        output_dir: 输出目录
        num_images: 要保存的图像数量（如果为None，则保存全部）
    """
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]  # 添加唯一标识符防止冲突
    output_dir = os.path.join(output_dir, f"cifar10_real_{timestamp}_{unique_id}")
    os.makedirs(output_dir, exist_ok = True)

    # CIFAR-10数据集的标准变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为[0,1]范围的tensor
    ])

    # 加载CIFAR-10数据集
    dataset = torchvision.datasets.CIFAR10(
        root = "/data1/yangyanliang/data/",
        train = True,
        download = False,
        transform = transform
    )

    # 如果指定了数量，则只使用前num_images个样本
    if num_images is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(num_images, len(dataset))))

    print(f"Saving {len(dataset)} CIFAR-10 images to {output_dir}")

    # 保存所有图像
    for i, (image, label) in enumerate(tqdm(dataset, desc = "Saving images")):
        # 生成唯一文件路径，使用时间戳和唯一ID防止冲突
        timestamp_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_suffix = uuid.uuid4().hex[:8]
        save_path = os.path.join(output_dir, f"cifar10_{i:05d}_{timestamp_ms}_{unique_suffix}.png")

        # 直接保存图像，不进行额外的范围转换
        save_image_tensor(image, save_path)

    print(f"Saved {len(dataset)} images to {output_dir}")


def main():
    save_cifar10_images(output_dir = "/data1/yangyanliang/Flow-Matching/images_cifar10/", num_images = 30000)


if __name__ == "__main__":
    main()