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
import torch.distributed as dist


def batch_sample_simple_distributed(config_path: str, total_samples: int, save_path: str):
    """
    使用指定的配置和fm_scheduler.Sampler进行高效的分布式并行采样。
    核心逻辑：每个GPU独立处理一部分批次，实现真正意义上的并行计算。

    Args:
        config_path (str): YAML配置文件的路径。
        total_samples (int): 要生成的总图像数。
        save_path (str): 保存生成图像的目录。
    """
    # 1. 初始化分布式环境
    conf = load_config(config_path)
    device, local_rank = init_distributed()
    world_size = dist.get_world_size()

    if local_rank == 0:
        os.makedirs(save_path, exist_ok=True)
        print(f"使用 {world_size} 个GPU进行并行采样, 图像将保存至: {save_path}")
    dist.barrier()

    # 2. 实例化模型和VAE
    model = instantiate_from_config(conf.trainer.params.model)
    model = load_model_from_checkpoint(
        checkpoint_path=conf.sampling.model_path, model_instance=model,
        model_type='dit', device=device, use_ema=True
    )
    vae = load_model_from_checkpoint(
        model_type='autoencoderkl', device=device, checkpoint_path=conf.sampling.vae_path
    )
    model.eval()
    vae.eval()

    # 3. 准备采样参数
    sampler = Sampler(model)
    sampling_kwargs = conf.sampling
    batch_size = sampling_kwargs['shape'][0]

    # 4. 计算总批次数，并为每个GPU分配任务
    total_batches = (total_samples + batch_size - 1) // batch_size
    # 每个GPU处理的批次索引列表
    batches_to_process = [i for i in range(total_batches) if i % world_size == local_rank]

    # 5. 为每个GPU创建自己的进度条（只在主进程显示）
    pbar = tqdm(total=len(batches_to_process), desc=f"GPU {local_rank} 采样中", disable=(local_rank != 0))

    with torch.no_grad():
        # 每个GPU独立地、并行地处理自己负责的批次
        for batch_idx in batches_to_process:
            # 确定当前批次是否是最后一个，以及其实际大小
            is_last_batch = (batch_idx == total_batches - 1)
            samples_in_this_batch = (total_samples % batch_size) if (
                        is_last_batch and total_samples % batch_size != 0) else batch_size

            current_kwargs = sampling_kwargs.copy()
            # 确保批次大小正确
            if current_kwargs['shape'][0] != samples_in_this_batch:
                current_kwargs['shape'][0] = samples_in_this_batch

            # 执行采样
            latents = sampler.sample(**current_kwargs, disable_tqdm=True)
            images = vae.decode(latents).sample

            # 保存图像，使用全局索引命名
            for j in range(samples_in_this_batch):
                global_sample_idx = batch_idx * batch_size + j
                if global_sample_idx < total_samples:
                    img_tensor = images[j]
                    img_save_path = os.path.join(save_path, f"sample_{global_sample_idx:05d}.png")
                    show_tensor_image(img_tensor, save_path=img_save_path)
            
            pbar.update(1)

    # 所有进程工作完成后在此同步
    dist.barrier()

    if local_rank == 0:
        pbar.close()
        final_file_count = len(os.listdir(save_path))
        print(f"\n采样完成！总共请求 {total_samples} 张图像，实际生成 {final_file_count} 张已保存至 '{save_path}'.")



if __name__ == "__main__":
    CONFIG_PATH = "/data1/yangyanliang/Flow-Matching/configs/celebahq_dit.yaml"
    TOTAL_SAMPLES = 30000
    SAVE_PATH = "/data1/yangyanliang/Flow-Matching/celeba_samples"

    if "LOCAL_RANK" not in os.environ:
        print("错误：此脚本需要通过 torch.distributed.launch 或 torchrun 启动以进行并行采样。")
    else:
        print("--- 开始分布式批量采样任务 (简洁版) ---")
        batch_sample_simple_distributed(
            config_path = CONFIG_PATH,
            total_samples = TOTAL_SAMPLES,
            save_path = SAVE_PATH
        )
