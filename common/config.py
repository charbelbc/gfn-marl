import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:

    scenario_name: str = "simple_spread"
    num_agents: int = 3
    num_landmarks: int = 3
    episode_length: int = 25
    action_dim: int = 5
    batch_size: int = 32
    minibatch_size: int = 8
    use_rnn: float = False
    normalize_value: bool = False
    value_clipping: bool = False

    # PPO parameters
    gamma: float = 0.99
    gaelambda: float = 0.95
    ppo_epochs: int = 10
    eps_clip: float = 0.2
    lr: float = 0.0007
