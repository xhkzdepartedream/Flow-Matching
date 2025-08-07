import os

import torch
from diffusers import AutoencoderKL
from fm_scheduler.Sampler import Sampler
from utils import instantiate_from_config, show_tensor_image, load_model_from_checkpoint, load_config


def main():
    # 1. Load configuration
    conf = load_config("../configs/afhq.yaml")

    # 2. Instantiate model and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(conf.trainer.params.model)

    model = load_model_from_checkpoint(
        model_instance = model,
        checkpoint_path = conf.sampling.model_path,
        model_type = 'dit',
        device = device
    )
    vae = AutoencoderKL.from_pretrained("checkpoints/autoencoderkl/")
    vae = vae.to(device)

    model.eval()
    vae.eval()

    # 3. Instantiate sampler
    sampler = Sampler(model)

    # 4. Prepare for sampling
    sampling_kwargs = conf.sampling
    output_dir = sampling_kwargs.pop("output_dir")
    os.makedirs(output_dir, exist_ok = True)

    # 5. Generate samples
    print(
        f"Starting sampling with {sampling_kwargs.get('sampler_type', 'euler')} for {sampling_kwargs.get('n_steps', 'N/A')} steps...")
    latents = sampler.sample(**sampling_kwargs)
    images = vae.decode(latents).sample

    for i, image in enumerate(images):
        print(image.shape)
        show_tensor_image(image, save_path = os.path.join(output_dir, "sample{i}.png"))
    print("Sampling finished.")


if __name__ == "__main__":
    main()
