import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from trainer.AutoencoderKL_trainer import AutoencoderKL_trainer
from utils import init_distributed
from data_processing.init_dataset import transform, AFHQDataset
from torch.nn.parallel import DistributedDataParallel as DDP

device, local_rank = init_distributed()


def main():

    dir = "/data1/yangyanliang/.cache/kagglehub/datasets/dimensi0n/afhq-512/versions/1/"

    train_dataset = AFHQDataset(dir, transform = transform)

    dir = "../../.cache/kagglehub/datasets/dimensi0n/afhq-512/versions/1/"
L18: pretrained_model_name_or_path = "../../Diffusion-Model/autoencoderkl_success2.pth"

    # Initialize the trainer
    autoencoderkl_launcher = AutoencoderKL_trainer(
        dataset = train_dataset,
        title = 'afhq',
        pretrained_model_name_or_path = None
    )

    # Wrap the model with DDP for distributed training
    autoencoderkl_launcher.vae = DDP(autoencoderkl_launcher.vae, device_ids = [local_rank],
                                     output_device = local_rank, find_unused_parameters = False)

    # Define training parameters
    epochs = 10
    batch_size = 3
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
