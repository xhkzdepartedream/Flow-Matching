import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from utils import init_distributed, instantiate_from_config, load_config

def main():
    # 1. Load configuration
    conf = load_config("./configs/celebahq_dit.yaml")
    device, local_rank = init_distributed()
    trainer = instantiate_from_config(conf.trainer)

    if conf.checkpoint.load:
        trainer._load_checkpoint(conf.checkpoint.path, local_rank)

    trainer.train(ignore_labels = conf.trainer.params.ignore_labels, save_every = conf.checkpoint.save_every)

if __name__ == "__main__":
    main()