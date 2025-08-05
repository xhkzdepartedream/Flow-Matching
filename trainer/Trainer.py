import torch
from torch_ema import ExponentialMovingAverage
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils import init_distributed, instantiate_from_config, setup_logger
from typing import Optional

device, local_rank = init_distributed()


class Trainer:
    """
    A robust, modular trainer for diffusion models, with corrected learning rate scheduling.
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
            gradient_accumulation_steps: int = 1,
            warmup_steps: int = 1000,
            **kwargs
    ):
        self.scheduler_config = scheduler
        self.model_config = model
        self.dataset_config = dataset
        self.title = title
        self.base_lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps

        self.device = device
        self.local_rank = local_rank
        self.logger = setup_logger(log_dir = './logs', name = title)
        self.start_epoch = 1
        self.global_step = 0

    def _setup(self):
        """Deferred setup until training starts."""
        self.scheduler = instantiate_from_config(self.scheduler_config)
        self.model = instantiate_from_config(self.model_config).to(self.device)
        self.dataset = instantiate_from_config(self.dataset_config)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.base_lr, betas = (0.9, 0.999),
                                           weight_decay = 1e-2)

        self.datasampler = DistributedSampler(self.dataset, shuffle = True)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = False,
                                     num_workers = 8, pin_memory = True, sampler = self.datasampler)

        self.scaler = torch.amp.GradScaler()
        self.model = DDP(self.model, device_ids = [self.local_rank])
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay = 0.999)

        num_update_steps_per_epoch = len(self.dataloader) // self.gradient_accumulation_steps
        self.total_steps = self.n_epoch * num_update_steps_per_epoch
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max = self.total_steps - self.warmup_steps,
            eta_min = self.base_lr / 10.0
        )

    def train_step(self, x1: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device = self.device) * (1.0 - 1e-5) + 1e-5
        x0 = torch.randn_like(x1)
        xt, target = self.scheduler.prepare_training_pair(x1, t, x0)

        prediction = self.model(xt, t, labels) if labels is not None else self.model(xt, t)
        loss = torch.mean((prediction - target) ** 2)
        return loss

    def _adjust_learning_rate(self):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, (self.global_step + 1) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale
        else:
            self.lr_scheduler.step()

    def train(self, ignore_labels: bool, save_every: int = 10):
        self._setup()
        self.model.train()
        self.optimizer.zero_grad(set_to_none = True)

        for epoch in range(self.start_epoch, self.n_epoch + 1):
            self.datasampler.set_epoch(epoch)
            pbar = tqdm(self.dataloader, desc = f"Epoch {epoch}", disable = (self.local_rank != 0))
            epoch_total_loss = 0.0
            update_steps_in_epoch = 0

            for i, batch_data in enumerate(pbar):
                x1 = (batch_data[0] if ignore_labels else batch_data).to(self.device)
                labels = None  # Assuming unconditional for now

                with torch.amp.autocast('cuda'):
                    loss = self.train_step(x1, labels) / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                    self._adjust_learning_rate()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none = True)
                    self.ema.update()

                    self.global_step += 1
                    epoch_total_loss += loss.item() * self.gradient_accumulation_steps
                    update_steps_in_epoch += 1

                if self.local_rank == 0:
                    pbar.set_postfix(loss = loss.item() * self.gradient_accumulation_steps,
                                     lr = self.optimizer.param_groups[0]['lr'])

            avg_loss_on_rank = epoch_total_loss / update_steps_in_epoch if update_steps_in_epoch > 0 else 0.0
            avg_loss_tensor = torch.tensor(avg_loss_on_rank, device = self.device)
            torch.distributed.all_reduce(avg_loss_tensor, op = torch.distributed.ReduceOp.SUM)
            final_avg_loss = avg_loss_tensor.item() / torch.distributed.get_world_size()

            if self.local_rank == 0:
                self.logger.info(f"Epoch {epoch} | Average Loss: {final_avg_loss:.4f}")

            if epoch % save_every == 0:
                self._save_checkpoint(epoch)

        if self.local_rank == 0:
            self.logger.info("Training finished.")

    def _save_checkpoint(self, epoch: int):
        if self.local_rank != 0: return
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{self.title}_{epoch}.pth")
        self.logger.info(f"MODEL Checkpoint saved at epoch {epoch}.")

    def _load_checkpoint(self, checkpoint_path: str, rank: int, use_ema: bool = True):
        self._setup()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location = map_location)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if use_ema: self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.logger.info(f"[RANK {rank}] Loaded checkpoint from epoch {checkpoint['epoch']}.")
