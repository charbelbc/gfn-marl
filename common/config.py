import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:

    scenario_name: str = "simple_pp"
    total_steps: int = 5_000_000
    num_agents: int = 2
    num_landmarks: int = 3
    capture_requirement: int = 1
    episode_length: int = 70
    action_dim: int = 5
    batch_size: int = 64
    minibatch_size: int = 64
    use_rnn: float = True
    normalize_value: bool = True
    value_clipping: bool = False
    reward_normalization: bool = False
    obs_dim: int = 37
    state_dim: int = num_agents * obs_dim

    actor: Dict = field(
        default_factory=lambda: {
            "no_memory_fc_layers": [64, 64],
            "in_fc_layers": [64, 64],
            "memory_size": 64,
            "activation": torch.nn.Tanh,
            "use_feature_norm": True,
            "use_layer_norm": True,
            "out_fc_layers": [],
            "orthogonal_init": True,
        }
    )

    critic: Dict = field(
        default_factory=lambda: {
            "no_memory_fc_layers": [64, 64],
            "in_fc_layers": [64, 64],
            "memory_size": 64,
            "activation": torch.nn.Tanh,
            "use_feature_norm": True,
            "use_layer_norm": True,
            "out_fc_layers": [],
            "orthogonal_init": True,
        }
    )

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
    gfn_ar_policy: bool = True
    gfn_single_codebook: bool = True
    gfn_sampling_exponent: float = 1
    gfn_num_samples: int = 2
    gfn_encoder_steps: int = 10
    gfn_decoder_steps: int = 1
