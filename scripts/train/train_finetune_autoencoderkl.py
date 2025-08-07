import os

import torch.utils.data.dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from trainer.AutoencoderKL_trainer import AutoencoderKL_trainer
from utils import init_distributed
from data_processing.init_dataset import transform, CelebaHQDataset
from torch.nn.parallel import DistributedDataParallel as DDP

device, local_rank = init_distributed()


def main():

    dir = "/data1/yangyanliang/.cache/kagglehub/datasets/chelove4draste/ffhq-256x256/versions/1/ffhq256/"

    train_dataset = CelebaHQDataset(dir, transform = transform)

    pretrained_model_name_or_path = "/data1/yangyanliang/autoencoderkl_finetuned_celeba_hq2/"

    # Initialize the trainer
    autoencoderkl_launcher = AutoencoderKL_trainer(
        dataset = train_dataset,
        title = 'unified',
        pretrained_model_name_or_path = pretrained_model_name_or_path
    )

    # Wrap the model with DDP for distributed training
    autoencoderkl_launcher.vae = DDP(autoencoderkl_launcher.vae, device_ids = [local_rank],
                                     output_device = local_rank, find_unused_parameters = False)

    # Define training parameters
    epochs = 1
    batch_size = 2
    lr = 1e-4  # Fine-tuning typically uses a smaller learning rate
    recon_factor = 1.0
    kl_factor = 1e-6
    perc_factor = 0.3

    # Start training
    autoencoderkl_launcher.train(
        epochs = epochs,
        batch_size = batch_size,
        lr = lr,
        recon_factor = recon_factor,
        kl_factor = kl_factor,
        perc_factor = perc_factor,
    )


if __name__ == '__main__':
    main()
