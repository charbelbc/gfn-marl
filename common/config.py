import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:

    scenario_name: str = "simple_spread"
    num_agents: int = 4
    num_landmarks: int = 3
    capture_requirement: int = 2
    episode_length: int = 35
    action_dim: int = 5
    batch_size: int = 64
    minibatch_size: int = 64
    use_rnn: float = True
    normalize_value: bool = True
    value_clipping: bool = False
    reward_normalization: bool = False
    obs_dim: int = 37
    state_dim: int = num_agents * obs_dim

    # PPO parameters
    gamma: float = 0.99
    gaelambda: float = 0.95
    ppo_epochs: int = 15
    eps_clip: float = 0.2
    lr: float = 0.0005

    use_gfn: bool = True
    gfn_dict_size: int = 3
    gfn_state_size: int = 3
    gfn_lr: float = 0.0005
    gfn_logz_lr: float = 0.1
    gfn_dec_lr: float = 0.0005
    gfn_rand_prob: float = 0.3
    gfn_greedy_decoder: bool = True
    gfn_use_pb: bool = False
    gfn_single_codebook: bool = True
    gfn_sampling_exponent: float = -1
    gfn_encoder_steps: int = 20
    gfn_decoder_steps: int = 20
