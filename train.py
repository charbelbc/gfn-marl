import torch
import numpy as np
from common.config import Config
import gymnasium as gym
from alg.mappo import MPE_GFN_MAPPO, MPE_MAPPO
from common.utils import ParallelEnv, MPE_ReplayBuffer, my_f
from mpe.MPE_env import MPEEnv
from multiagent.make_env import make_env
from common.utils import Normalization

import multigrid.envs
import gymnasium as gym
import minigrid
from multigrid.wrappers import FullyObsWrapper

import wandb
import os


def train_mpe(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)

    batch_size = config.batch_size
    env = ParallelEnv(my_f, config, batch_size)
    config.obs_dim = env.obs_dim
    config.state_dim = env.obs_dim * config.num_agents
    agent = MPE_MAPPO(device=device, config=config)
    buffer = MPE_ReplayBuffer(config=config)

    episode = 0
    if config.reward_normalization:
        reward_norm = Normalization(config.num_agents)

    while (episode * config.episode_length) < 5_000_000:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        # doness = dones.clone()
        step = 0
        curr_reward = 0.0
        if config.use_rnn:
            actor_memory = torch.zeros(
                batch_size, config.num_agents, config.actor["memory_size"]
            ).to(device)
            critic_memory = torch.zeros(
                batch_size, config.num_agents, config.critic["memory_size"]
            ).to(device)

        for _ in range(config.episode_length):
            if config.use_rnn:
                actions, logits, value, actor_memory, critic_memory = (
                    agent.select_action(obs, actor_memory, critic_memory)
                )
            else:
                actions, logits, value = agent.select_action(obs)
            next_obs = env.step(actions.cpu())
            rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])
            if config.reward_normalization:
                normalized_rewards = reward_norm(rewards)
            dones = torch.stack([torch.tensor(o[2]).squeeze() for o in next_obs])
            buffer.store_transition(
                step,
                obs,
                actions.cpu(),
                logits.cpu(),
                value.squeeze().cpu(),
                (
                    normalized_rewards.squeeze()
                    if config.reward_normalization
                    else rewards.squeeze()
                ),
                dones,
            )
            curr_reward += rewards[:, 0].mean().item()
            obs = [o[0] for o in next_obs]
            step += 1

        if config.use_rnn:
            _, _, value, _, _ = agent.select_action(obs, actor_memory, critic_memory)
        else:
            _, _, value = agent.select_action(obs)
        buffer.buffer["state_values"][:, -1] = value.squeeze().cpu()

        loss_dict = agent.update(buffer)
        buffer.reset_buffer()
        episode += batch_size

        lr_now = config.lr * (1 - (episode * config.episode_length) / 20_000_000)
        agent.optimizer.param_groups[0]["lr"] = lr_now

        if logging:
            loss_dict.update({"reward": curr_reward})
            if episode % (50 * batch_size) == 0:
                agent.save_model(model_dir, episode * config.episode_length)
            wandb.log(
                loss_dict,
                step=episode * config.episode_length,
            )


def train_mpe_gfn(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)

    batch_size = config.batch_size
    env = ParallelEnv(my_f, config, batch_size)
    config.obs_dim = env.obs_dim
    config.state_dim = env.obs_dim * config.num_agents
    agent = MPE_GFN_MAPPO(device=device, config=config)
    buffer = MPE_ReplayBuffer(config=config)

    episode = 0
    if config.reward_normalization:
        reward_norm = Normalization(config.num_agents)

    while (episode * config.episode_length) < 5_000_000:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        step = 0
        curr_reward = 0.0
        if config.use_rnn:
            actor_memory = torch.zeros(
                batch_size, config.num_agents, config.actor["memory_size"]
            ).to(device)
            critic_memory = torch.zeros(
                batch_size, config.num_agents, config.critic["memory_size"]
            ).to(device)
        gfn_memory = torch.zeros(
            batch_size, config.num_agents, (config.num_agents - 1), 64
        ).to(agent.device)

        for _ in range(config.episode_length):
            if step == 0:
                actions = torch.zeros(batch_size, config.num_agents)
                rewards = torch.zeros(batch_size, config.num_agents)
            if config.use_rnn:
                (
                    actions,
                    logits,
                    value,
                    actor_memory,
                    critic_memory,
                    latents,
                    gfn_memory,
                ) = agent.select_action(
                    obs, actions, rewards, gfn_memory, actor_memory, critic_memory
                )
            else:
                actions, logits, value = agent.select_action(obs)
            next_obs = env.step(actions.cpu())
            rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])
            if config.reward_normalization:
                normalized_rewards = reward_norm(rewards)
            dones = torch.stack([torch.tensor(o[2]).squeeze() for o in next_obs])
            buffer.store_transition(
                step,
                obs,
                actions.cpu(),
                logits.cpu(),
                value.squeeze().cpu(),
                (
                    normalized_rewards.squeeze()
                    if config.reward_normalization
                    else rewards.squeeze()
                ),
                dones,
                latents.cpu(),
            )
            curr_reward += rewards[:, 0].mean().item()
            obs = [o[0] for o in next_obs]
            step += 1

        if config.use_rnn:
            _, _, value, _, _, _, _ = agent.select_action(
                obs, actions, rewards, gfn_memory, actor_memory, critic_memory
            )
        else:
            _, _, value = agent.select_action(obs)
        buffer.buffer["state_values"][:, -1] = value.squeeze().cpu()

        loss_dict = agent.update(buffer)
        gfn_loss_dict = agent.gflownet.update(buffer)
        loss_dict.update(gfn_loss_dict)
        buffer.reset_buffer()
        episode += batch_size

        lr_now = config.lr * (1 - (episode * config.episode_length) / 20_000_000)
        agent.optimizer.param_groups[0]["lr"] = lr_now

        if logging:
            loss_dict.update({"reward": curr_reward})
            if episode % (50 * batch_size) == 0:
                agent.save_model(model_dir, episode * config.episode_length)
            wandb.log(
                loss_dict,
                step=episode * config.episode_length,
            )
