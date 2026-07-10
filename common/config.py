import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict
from enum import Enum


@dataclass
class EnvConfig:
    pass


@dataclass
class SpreadConfig(EnvConfig):
    scenario_name: str = "simple_cn"
    num_agents: int = 3
    num_landmarks: int = 3
    episode_length: int = 25
    action_dim: int = 5


@dataclass
class PredatorPreyConfig(EnvConfig):
    scenario_name: str = "simple_pp"
    num_agents: int = 4
    capture_requirement: int = 2
    episode_length: int = 70
    action_dim: int = 5


@dataclass
class NetworkConfig:
    pass


@dataclass
class MLPConfig(NetworkConfig):
    no_memory_fc_layers: list[int] = field(default_factory=lambda: [64, 64])
    in_fc_layers: list[int] = field(default_factory=lambda: [64, 64])
    memory_size: int = 64
    activation: type = torch.nn.Tanh
    use_feature_norm: bool = True
    use_layer_norm: bool = True
    out_fc_layers: list[int] = field(default_factory=list)
    orthogonal_init: bool = True


@dataclass
class TrainingConfig:
    total_steps: int = 5_000_000
    batch_size: int = 64
    minibatch_size: int = 64
    use_rnn: float = True
    normalize_value: bool = True
    value_clipping: bool = False
    reward_normalization: bool = False
    actor: NetworkConfig = field(default_factory=MLPConfig)
    critic: NetworkConfig = field(default_factory=MLPConfig)


@dataclass
class AlgorithmConfig:
    pass


@dataclass
class PPOConfig(AlgorithmConfig):
    gamma: float = 0.99
    gaelambda: float = 0.95
    ppo_epochs: int = 15
    eps_clip: float = 0.2
    lr: float = 0.0005


class ModuleType(Enum):
    NONE = "none"
    GFN = "gfn"
    VAE = "vae"
    VQVAE = "vqvae"
    SUPTOM = "suptom"


@dataclass
class ModuleConfig:
    type: ModuleType = ModuleType.NONE


@dataclass
class GFlowNetConfig(ModuleConfig):
    type: ModuleType = ModuleType.GFN
    gfn_dict_size: int = 3
    gfn_state_size: int = 3
    gfn_lr: float = 0.0005
    gfn_logz_lr: float = 0.1
    gfn_dec_lr: float = 0.0005
    gfn_rand_prob: float = 0.3
    gfn_greedy_decoder: bool = True
    gfn_use_pb: bool = True
    gfn_ar_policy: bool = False
    gfn_single_codebook: bool = True
    gfn_sampling_exponent: float = 1
    gfn_num_samples: int = 2
    gfn_encoder_steps: int = 10
    gfn_decoder_steps: int = 1


@dataclass
class VQVAEConfig(ModuleConfig):
    type: ModuleType = ModuleType.VQVAE
    vqvae_dict_size: int = 3
    vqvae_state_size: int = 3
    vqvae_lr: float = 0.0005


@dataclass
class VAEConfig(ModuleConfig):
    type: ModuleType = ModuleType.VAE
    vae_latent_size: int = 3
    vae_kl_factor: float = 0.05
    vae_lr: float = 0.0005


@dataclass
class SupTomConfig(ModuleConfig):
    type: ModuleType = ModuleType.SUPTOM
    goal_size: list[int] = field(default_factory=lambda: [2, 2])
    belief_lr: float = 0.0005
    belief_reward_factor: float = 1.0


@dataclass
class Config:

    env: EnvConfig = field(default_factory=SpreadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alg: AlgorithmConfig = field(default_factory=PPOConfig)
    module: ModuleConfig = field(default_factory=GFlowNetConfig)

    # scenario_name: str = "simple_pp"
    # total_steps: int = 5_000_000
    # num_agents: int = 2
    # num_landmarks: int = 3
    # capture_requirement: int = 1
    # episode_length: int = 70
    # action_dim: int = 5

    # batch_size: int = 64
    # minibatch_size: int = 64
    # use_rnn: float = True
    # normalize_value: bool = True
    # value_clipping: bool = False
    # reward_normalization: bool = False
    # obs_dim: int = 37
    # state_dim: int = num_agents * obs_dim

    # actor: Dict = field(
    #     default_factory=lambda: {
    #         "no_memory_fc_layers": [64, 64],
    #         "in_fc_layers": [64, 64],
    #         "memory_size": 64,
    #         "activation": torch.nn.Tanh,
    #         "use_feature_norm": True,
    #         "use_layer_norm": True,
    #         "out_fc_layers": [],
    #         "orthogonal_init": True,
    #     }
    # )

    # critic: Dict = field(
    #     default_factory=lambda: {
    #         "no_memory_fc_layers": [64, 64],
    #         "in_fc_layers": [64, 64],
    #         "memory_size": 64,
    #         "activation": torch.nn.Tanh,
    #         "use_feature_norm": True,
    #         "use_layer_norm": True,
    #         "out_fc_layers": [],
    #         "orthogonal_init": True,
    #     }
    # )

    # # PPO parameters
    # gamma: float = 0.99
    # gaelambda: float = 0.95
    # ppo_epochs: int = 15
    # eps_clip: float = 0.2
    # lr: float = 0.0005

    # use_gfn: bool = True
    # gfn_dict_size: int = 3
    # gfn_state_size: int = 3
    # gfn_lr: float = 0.0005
    # gfn_logz_lr: float = 0.1
    # gfn_dec_lr: float = 0.0005
    # gfn_rand_prob: float = 0.3
    # gfn_greedy_decoder: bool = True
    # gfn_use_pb: bool = True
    # gfn_ar_policy: bool = False
    # gfn_single_codebook: bool = True
    # gfn_sampling_exponent: float = 1
    # gfn_num_samples: int = 2
    # gfn_encoder_steps: int = 10
    # gfn_decoder_steps: int = 1

    # use_vqvae: bool = True
    # vqvae_dict_size: int = 3
    # vqvae_state_size: int = 3
    # vqvae_lr: float = 0.0005

    # use_vae: bool = True
    # vae_latent_size: int = 3
    # vae_kl_factor: float = 0.05
    # vae_lr: float = 0.0005
