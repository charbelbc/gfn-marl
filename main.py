import numpy as np
import gymnasium as gym
from common.config import Config
from train import train, train_mpe, train_mpe_single
import os
import wandb
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():

    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)

    config = Config()

    with wandb.init(project="wi-symmetry", config=config.__dict__):
        train_mpe(config=config)


if __name__ == "__main__":
    main()
