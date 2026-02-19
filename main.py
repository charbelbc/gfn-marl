import numpy as np
import gymnasium as gym
from common.config import Config
from train import train
import os
import wandb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():

    wandb.login(key="12fcfca1020dbadb9edb283382ef69da389c17eb")

    config = Config()

    with wandb.init(project="wi-symmetry", config=config.__dict__):
        train(config=config)


if __name__ == "__main__":
    main()
