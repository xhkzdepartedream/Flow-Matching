import torch
import os
from tqdm import tqdm
from utils import (
    load_config,
    instantiate_from_config,
    load_model_from_checkpoint,
    show_tensor_image,
    init_distributed
)
from fm_scheduler.Sampler import Sampler

def batch_sample(config_path: str, total_samples: int, save_path: str):
    """
    使用指定的配置和fm_scheduler.Sampler进行批量采样。

    Args:
        config_path (str): YAML配置文件的路径。
        total_samples (int): 要生成的总图像数。
        save_path (str): 保存生成图像的目录。
    """
    # 1. 加载配置并设置设备
    conf = load_config(config_path)
    device, _ = init_distributed()
    os.makedirs(save_path, exist_ok = True)
    print(f"使用设备: {device}, 图像将保存至: {save_path}")

    # 2. 实例化模型和VAE并加载权重
    model = instantiate_from_config(conf.trainer.params.model)
    model = load_model_from_checkpoint(
        checkpoint_path = conf.sampling.model_path,
        model_instance = model,
        model_type = 'dit',
        device = device,
        use_ema = True
    )
    vae = load_model_from_checkpoint(
        model_type = 'autoencoderkl',
        device = device,
        checkpoint_path = conf.sampling.vae_path
    )
    model.eval()
    vae.eval()

    # 3. 实例化采样器
    sampler = Sampler(model)
    sampling_kwargs = conf.sampling
    batch_size = sampling_kwargs.get("batch_size", 1)

    # 4. 开始批量采样
    pbar = tqdm(total=total_samples, desc="正在生成图像", position=0)
    saved_count = 0
    with torch.no_grad():
        while saved_count < total_samples:
            # 计算当前批次的大小
            current_batch_size = min(batch_size, total_samples - saved_count)
            if current_batch_size <= 0:
                break

            # 更新采样参数中的批次大小
            sampling_kwargs["batch_size"] = current_batch_size

            # 生成潜在向量和图像
            latents = sampler.sample(**sampling_kwargs)
            images = vae.decode(latents).sample

            # 保存图像
            for i in range(current_batch_size):
                img_tensor = images[i]
                img_save_path = os.path.join(save_path, f"sample_{saved_count:05d}.png")
                show_tensor_image(img_tensor, save_path = img_save_path)
                saved_count += 1
                pbar.update(1)

    pbar.close()
    print(f"\n采样完成！总共 {saved_count} 张图像已保存至 '{save_path}'.")


if __name__ == "__main__":
    # --- 在此直接设置参数 ---
    CONFIG_PATH = "../configs/celebahq_dit.yaml"
    TOTAL_SAMPLES = 30000
    SAVE_PATH = "/data1/yangyanliang/Flow-Matching/celeba_samples"
    # -------------------------

    print("--- 开始批量采样任务 ---")
    print(f"配置文件: {CONFIG_PATH}")
    print(f"采样总数: {TOTAL_SAMPLES}")
    print(f"保存路径: {SAVE_PATH}")
    print("-------------------------")

    batch_sample(
        config_path = CONFIG_PATH,
        total_samples = TOTAL_SAMPLES,
        save_path = SAVE_PATH
    )
