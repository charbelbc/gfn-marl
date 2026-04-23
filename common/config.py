import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:

    scenario_name: str = "simple_spread"
    num_agents: int = 2
    num_landmarks: int = 2
    episode_length: int = 50

    env_name: str = "MultiGrid-TwoTasksEnv-v0"
    training_steps: int = 10 * 10**6
    model_type: str = "standard"

    # Environment parameters
    n_agents: int = 2
    action_dim: int = 5
    max_steps: int = 50
    batch_size: int = 20

    # Models parameters
    memory_size: int = 512

    # PPO parameters
    gamma: float = 0.99
    gaelambda: float = 0.95
    ppo_epochs: int = 10
    eps_clip: float = 0.2
    lr: float = 0.0005
