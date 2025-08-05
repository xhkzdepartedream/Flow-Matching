import torch
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils import init_distributed, instantiate_from_config, setup_logger
from typing import Optional

device, local_rank = init_distributed()


class Trainer:
    """
    A unified, modular trainer for diffusion models, adapted to the project's style.
    It handles the training loop, distributed data_processing parallel setup, gradient scaling,
    and checkpointing in a generic way.
    """

    def __init__(
            self,
            scheduler: dict,
            model: dict,
            dataset: Dataset,
            title: str,
            lr: float,
            n_epoch: int,
            batch_size: int,
            **kwargs
    ):
        # 实例化调度器和模型
        self.scheduler = instantiate_from_config(scheduler)
        self.model = instantiate_from_config(model).to(device)

        self.device = device
        self.dataset = instantiate_from_config(dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, betas = (0.99, 0.999), weight_decay = 0)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.title = title
        self.losses = []
        self.model_conditional = self.model.has_cond if hasattr(self.model, 'has_cond') else False
        self.start_epoch = 1

        self.logger = setup_logger(log_dir = './logs', name = title)

        self.datasampler = DistributedSampler(self.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = False,
                                     num_workers = 32, pin_memory = True, sampler = self.datasampler)
        self.scaler = torch.amp.GradScaler()
        self.device = device
        self.local_rank = local_rank

        self.scheduler_lrs = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = 20,
            T_mult = 1,
            eta_min = 1e-5
        )
        self.model = DDP(self.model, device_ids = [self.local_rank])

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay = 0.999)

    def train_step(self, x1: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行一个训练步骤
        
        Args:
            x1: 真实数据样本
            labels: 条件标签（如果有）
            
        Returns:
            loss: 计算得到的损失值
        """
        batch_size = x1.shape[0]

        # 采样时间步
        t = torch.rand(batch_size, device = self.device) * (1.0 - 1e-5) + 1e-5

        # 采样噪声
        x0 = torch.randn_like(x1)

        # 通过调度器计算输入和目标
        xt, target = self.scheduler.prepare_training_pair(x1, t, x0)

        # 模型前向传播
        if labels is not None:
            prediction = self.model(xt, t, labels)
        else:
            prediction = self.model(xt, t)

        # 计算损失
        loss = torch.mean((prediction - target) ** 2)

        return loss

    def _save_checkpoint(self, epoch: int):
        """Saves a checkpoint of the model and optimizer states."""
        if self.local_rank != 0:
            return

        model_state = self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler_lrs.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
        }
        torch.save(checkpoint, f"{self.title}_{epoch}.pth")
        self.logger.info(f"Flow-Matching MODEL Checkpoint saved at epoch {epoch}.")

    def _load_checkpoint(self, checkpoint_path: str, rank: int, use_ema: bool = True):
        self.checkpoint_path = checkpoint_path
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # 把保存时的 0号GPU 映射到当前rank

        checkpoint = torch.load(checkpoint_path, map_location = map_location)

        if use_ema and 'ema_state_dict' in checkpoint:
            # 加载EMA权重到模型
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            # 将EMA权重复制到模型参数
            self.ema.copy_to()
            self.logger.info(f"[RANK {rank}] Loaded EMA checkpoint from epoch {checkpoint['epoch']}.")
        else:
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler_lrs.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if not use_ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            
        self.start_epoch = checkpoint['epoch'] + 1

        self.logger.info(f"[RANK {rank}] Loaded checkpoint from epoch {checkpoint['epoch']}.")

    def train(self, ignore_labels: bool, save_every: int = 10):
        # 合并条件和无条件训练逻辑
        for epoch in range(self.start_epoch, self.n_epoch + 1):
            self.datasampler.set_epoch(epoch)
            total_loss = 0
            pbar = tqdm(self.dataloader, desc = f"Epoch {epoch}", disable = (self.local_rank != 0))

            for batch_data in pbar:
                # 根据是否为条件模型处理数据
                if self.model_conditional:
                    batch, label = batch_data
                    x1 = batch.to(self.device)
                    labels = label
                else:
                    if ignore_labels:
                        x1, _ = batch_data
                    else:
                        x1 = batch_data
                    x1 = x1.to(self.device)
                    labels = None

                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self.train_step(x1, labels)

                # Additional NaN detection after loss computation
                if torch.isnan(loss):
                    self.logger.error(f"NaN loss detected at epoch {epoch}")
                    self.logger.error(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
                    self.logger.error(f"Scaler scale: {self.scaler.get_scale()}")
                    # Check model parameters for NaN
                    for name, param in self.model.named_parameters():
                        if torch.any(torch.isnan(param)):
                            self.logger.error(f"NaN in parameter: {name}")
                        if param.grad is not None and torch.any(torch.isnan(param.grad)):
                            self.logger.error(f"NaN in gradient: {name}")
                    raise RuntimeError(f"NaN loss at epoch {epoch}")

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update()
                self.scheduler_lrs.step(epoch + len(self.losses) / len(self.dataloader))
                total_loss += loss.item()

                if self.local_rank == 0:
                    pbar.set_postfix(loss = loss.item())

            self.losses.append(total_loss)
            if self.local_rank == 0:
                self.logger.info(f"Epoch {epoch} | Loss: {total_loss:.4f}")

            if epoch % save_every == 0:
                self._save_checkpoint(epoch)

        if self.local_rank == 0:
            self.logger.info("Training finished.")
