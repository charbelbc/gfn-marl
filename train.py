import torch
import numpy as np
from common.config import Config
import gymnasium as gym
from alg.mappo import (
    MPE_GFN_MAPPO,
    MPE_MAPPO,
    MPE_VQVAE_MAPPO,
    MPE_VAE_MAPPO,
    MPE_SUPTOM_MAPPO,
)
from common.utils import ParallelEnv, MPE_ReplayBuffer, my_f
from env.multiagent.make_env import make_env
from common.utils import Normalization

import env.multigrid.envs
import gymnasium as gym
import minigrid
from env.multigrid.wrappers import FullyObsWrapper

import wandb
import os


def train_mpe(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)
    seed = np.random.randint(1_000_000)

    batch_size = config.training.batch_size
    env = ParallelEnv(my_f, config, batch_size, seed)
    config.seed = seed
    config.env.obs_dim = env.obs_dim
    config.env.state_dim = env.obs_dim * config.env.num_agents
    agent = MPE_MAPPO(device=device, config=config)
    buffer = MPE_ReplayBuffer(config=config)

    episode = 0
    if config.training.reward_normalization:
        reward_norm = Normalization(config.env.num_agents)

    while (episode * config.env.episode_length) < config.training.total_steps:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        # doness = dones.clone()
        step = 0
        curr_reward = 0.0
        if config.training.use_rnn:
            actor_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.actor.memory_size
            ).to(device)
            critic_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.critic.memory_size
            ).to(device)

        for _ in range(config.env.episode_length):
            if config.training.use_rnn:
                actions, logits, value, actor_memory, critic_memory = (
                    agent.select_action(obs, actor_memory, critic_memory)
                )
            else:
                actions, logits, value = agent.select_action(obs)
            next_obs = env.step(actions.cpu())
            rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])
            if config.training.reward_normalization:
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
                    if config.training.reward_normalization
                    else rewards.squeeze()
                ),
                dones,
            )
            curr_reward += rewards[:, 0].mean().item()
            obs = [o[0] for o in next_obs]
            step += 1

        if config.training.use_rnn:
            _, _, value, _, _ = agent.select_action(obs, actor_memory, critic_memory)
        else:
            _, _, value = agent.select_action(obs)
        buffer.buffer["state_values"][:, -1] = value.squeeze().cpu()

        loss_dict = agent.update(buffer)
        buffer.reset_buffer()
        episode += batch_size

        lr_now = config.alg.lr * (
            1 - (episode * config.env.episode_length) / 20_000_000
        )
        agent.optimizer.param_groups[0]["lr"] = lr_now

        if logging:
            loss_dict.update({"reward": curr_reward})
            if episode % (50 * batch_size) == 0:
                agent.save_model(model_dir, episode * config.env.episode_length)
            wandb.log(
                loss_dict,
                step=episode * config.env.episode_length,
            )


def train_mpe_gfn(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)
    seed = np.random.randint(1_000_000)

    batch_size = config.training.batch_size
    env = ParallelEnv(my_f, config, batch_size)
    config.seed = seed
    config.env.obs_dim = env.obs_dim
    config.env.state_dim = env.obs_dim * config.env.num_agents
    agent = MPE_GFN_MAPPO(device=device, config=config)
    buffer = MPE_ReplayBuffer(config=config)

    episode = 0
    if config.training.reward_normalization:
        reward_norm = Normalization(config.env.num_agents)

    while (episode * config.env.episode_length) < config.training.total_steps:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        step = 0
        curr_reward = 0.0
        if config.training.use_rnn:
            actor_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.actor.memory_size
            ).to(device)
            critic_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.critic.memory_size
            ).to(device)
        gfn_memory = torch.zeros(
            batch_size, config.env.num_agents, (config.env.num_agents - 1), 64
        ).to(agent.device)

        for _ in range(config.env.episode_length):
            if step == 0:
                actions = torch.zeros(batch_size, config.env.num_agents)
                rewards = torch.zeros(batch_size, config.env.num_agents)
            if config.training.use_rnn:
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
                actions, logits, value, latents, gfn_memory = agent.select_action(
                    obs, actions, rewards, gfn_memory
                )
            next_obs = env.step(actions.cpu())
            rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])
            if config.training.reward_normalization:
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
                    if config.training.reward_normalization
                    else rewards.squeeze()
                ),
                dones,
                latents.cpu(),
            )
            curr_reward += rewards[:, 0].mean().item()
            obs = [o[0] for o in next_obs]
            step += 1

        if config.training.use_rnn:
            _, _, value, _, _, _, _ = agent.select_action(
                obs, actions, rewards, gfn_memory, actor_memory, critic_memory
            )
        else:
            _, _, value, _, _ = agent.select_action(obs, actions, rewards, gfn_memory)
        buffer.buffer["state_values"][:, -1] = value.squeeze().cpu()

        loss_dict = agent.update(buffer)
        gfn_loss_dict = agent.gflownet.update(buffer)
        loss_dict.update(gfn_loss_dict)
        buffer.reset_buffer()
        episode += batch_size

        lr_now = config.alg.lr * (
            1 - (episode * config.env.episode_length) / 20_000_000
        )
        agent.optimizer.param_groups[0]["lr"] = lr_now

        print("xxx")

        if logging:
            loss_dict.update({"reward": curr_reward})
            if episode % (50 * batch_size) == 0:
                agent.save_model(model_dir, episode * config.env.episode_length)
            wandb.log(
                loss_dict,
                step=episode * config.env.episode_length,
            )


def train_mpe_vqvae(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)
    seed = np.random.randint(1_000_000)

    batch_size = config.training.batch_size
    env = ParallelEnv(my_f, config, batch_size)
    config.seed = seed
    config.env.obs_dim = env.obs_dim
    config.env.state_dim = env.obs_dim * config.env.num_agents
    agent = MPE_VQVAE_MAPPO(device=device, config=config)
    buffer = MPE_ReplayBuffer(config=config)

    episode = 0
    if config.training.reward_normalization:
        reward_norm = Normalization(config.env.num_agents)

    while (episode * config.env.episode_length) < config.training.total_steps:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        step = 0
        curr_reward = 0.0
        if config.training.use_rnn:
            actor_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.actor.memory_size
            ).to(device)
            critic_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.critic.memory_size
            ).to(device)
        vqvae_memory = torch.zeros(
            batch_size, config.env.num_agents, (config.env.num_agents - 1), 64
        ).to(agent.device)

        for _ in range(config.env.episode_length):
            if step == 0:
                actions = torch.zeros(batch_size, config.env.num_agents)
                rewards = torch.zeros(batch_size, config.env.num_agents)
            if config.training.use_rnn:
                (
                    actions,
                    logits,
                    value,
                    actor_memory,
                    critic_memory,
                    latents,
                    vqvae_memory,
                ) = agent.select_action(
                    obs, actions, rewards, vqvae_memory, actor_memory, critic_memory
                )
            else:
                actions, logits, value, latents, vqvae_memory = agent.select_action(
                    obs, actions, rewards, vqvae_memory
                )
            next_obs = env.step(actions.cpu())
            rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])
            if config.training.reward_normalization:
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
                    if config.training.reward_normalization
                    else rewards.squeeze()
                ),
                dones,
                latents.cpu(),
            )
            curr_reward += rewards[:, 0].mean().item()
            obs = [o[0] for o in next_obs]
            step += 1

        if config.training.use_rnn:
            _, _, value, _, _, _, _ = agent.select_action(
                obs, actions, rewards, vqvae_memory, actor_memory, critic_memory
            )
        else:
            _, _, value, _, _ = agent.select_action(obs, actions, rewards, vqvae_memory)
        buffer.buffer["state_values"][:, -1] = value.squeeze().cpu()

        loss_dict = agent.update(buffer)
        vqvae_loss_dict = agent.vqvae.update(buffer)
        loss_dict.update(vqvae_loss_dict)
        buffer.reset_buffer()
        episode += batch_size

        lr_now = config.alg.lr * (
            1 - (episode * config.env.episode_length) / 20_000_000
        )
        agent.optimizer.param_groups[0]["lr"] = lr_now

        if logging:
            loss_dict.update({"reward": curr_reward})
            if episode % (50 * batch_size) == 0:
                agent.save_model(model_dir, episode * config.env.episode_length)
            wandb.log(
                loss_dict,
                step=episode * config.env.episode_length,
            )


def train_mpe_vae(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)
    seed = np.random.randint(1_000_000)

    batch_size = config.training.batch_size
    env = ParallelEnv(my_f, config, batch_size)
    config.seed = seed
    config.env.obs_dim = env.obs_dim
    config.env.state_dim = env.obs_dim * config.env.num_agents
    agent = MPE_VAE_MAPPO(device=device, config=config)
    buffer = MPE_ReplayBuffer(config=config)

    episode = 0
    if config.training.reward_normalization:
        reward_norm = Normalization(config.env.num_agents)

    while (episode * config.env.episode_length) < config.training.total_steps:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        step = 0
        curr_reward = 0.0
        if config.training.use_rnn:
            actor_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.actor.memory_size
            ).to(device)
            critic_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.critic.memory_size
            ).to(device)
        vae_memory = torch.zeros(
            batch_size, config.env.num_agents, (config.env.num_agents - 1), 64
        ).to(agent.device)

        for _ in range(config.env.episode_length):
            if step == 0:
                actions = torch.zeros(batch_size, config.env.num_agents)
                rewards = torch.zeros(batch_size, config.env.num_agents)
            if config.training.use_rnn:
                (
                    actions,
                    logits,
                    value,
                    actor_memory,
                    critic_memory,
                    latents,
                    vae_memory,
                ) = agent.select_action(
                    obs, actions, rewards, vae_memory, actor_memory, critic_memory
                )
            else:
                actions, logits, value, latents, vae_memory = agent.select_action(
                    obs, actions, rewards, vae_memory
                )
            next_obs = env.step(actions.cpu())
            rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])
            if config.training.reward_normalization:
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
                    if config.training.reward_normalization
                    else rewards.squeeze()
                ),
                dones,
                latents.cpu(),
            )
            curr_reward += rewards[:, 0].mean().item()
            obs = [o[0] for o in next_obs]
            step += 1

        if config.training.use_rnn:
            _, _, value, _, _, _, _ = agent.select_action(
                obs, actions, rewards, vae_memory, actor_memory, critic_memory
            )
        else:
            _, _, value, _, _ = agent.select_action(obs, actions, rewards, vae_memory)
        buffer.buffer["state_values"][:, -1] = value.squeeze().cpu()

        loss_dict = agent.update(buffer)
        vae_loss_dict = agent.vae.update(buffer)
        loss_dict.update(vae_loss_dict)
        buffer.reset_buffer()
        episode += batch_size

        lr_now = config.alg.lr * (
            1 - (episode * config.env.episode_length) / 20_000_000
        )
        agent.optimizer.param_groups[0]["lr"] = lr_now

        if logging:
            loss_dict.update({"reward": curr_reward})
            if episode % (50 * batch_size) == 0:
                agent.save_model(model_dir, episode * config.env.episode_length)
            wandb.log(
                loss_dict,
                step=episode * config.env.episode_length,
            )


def train_mpe_suptom(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    model_dir = os.path.join("saved_models")
    os.makedirs(model_dir, exist_ok=True)
    seed = np.random.randint(1_000_000)

    batch_size = config.training.batch_size
    env = ParallelEnv(my_f, config, batch_size)
    config.seed = seed
    config.env.obs_dim = env.obs_dim
    config.env.state_dim = env.obs_dim * config.env.num_agents
    agent = MPE_SUPTOM_MAPPO(device=device, config=config)
    buffer = MPE_ReplayBuffer(config=config)

    episode = 0
    if config.training.reward_normalization:
        reward_norm = Normalization(config.env.num_agents)

    while (episode * config.env.episode_length) < config.training.total_steps:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        step = 0
        curr_reward = 0.0
        if config.training.use_rnn:
            actor_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.actor.memory_size
            ).to(device)
            critic_memory = torch.zeros(
                batch_size, config.env.num_agents, config.training.critic.memory_size
            ).to(device)

        env_labels = []

        for _ in range(config.env.episode_length):
            if config.training.use_rnn:
                (
                    actions,
                    logits,
                    value,
                    actor_memory,
                    critic_memory,
                    beliefs,
                    others_beliefs,
                ) = agent.select_action(obs, actor_memory, critic_memory)
            else:
                actions, logits, value, beliefs, others_beliefs = agent.select_action(
                    obs
                )
            next_obs = env.step(actions.cpu())
            rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])

            # compute belief reward
            belief_targets = torch.empty_like(others_beliefs)
            for i in range(config.env.num_agents):
                other_agents = [j for j in range(config.env.num_agents) if j != i]
                belief_targets[:, i] = beliefs[:, other_agents].squeeze()

            size_start = 0
            for size in config.module.goal_size:
                pred_chunk = others_beliefs[..., size_start : size_start + size]
                target_chunk = belief_targets[..., size_start : size_start + size]
                belief_reward = (target_chunk * pred_chunk.log_softmax(dim=-1)).sum(
                    dim=(-1, -2)
                )
                size_start += size
            rewards += config.module.belief_reward_factor * belief_reward

            # collect env labels to train the beliefs
            env_labels.append(torch.ones((batch_size, sum(config.module.goal_size))))
            ##############

            if config.training.reward_normalization:
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
                    if config.training.reward_normalization
                    else rewards.squeeze()
                ),
                dones,
                # latents.cpu(),
            )
            curr_reward += rewards[:, 0].mean().item()
            obs = [o[0] for o in next_obs]
            step += 1

        if config.training.use_rnn:
            _, _, value, _, _, _, _ = agent.select_action(
                obs, actor_memory, critic_memory
            )
        else:
            _, _, value, _, _ = agent.select_action(obs)
        buffer.buffer["state_values"][:, -1] = value.squeeze().cpu()

        print("x")
        loss_dict = agent.update(buffer)

        env_labels = torch.stack(env_labels, dim=1)
        suptom_loss_dict = agent.actor.update(buffer, env_labels)
        loss_dict.update(suptom_loss_dict)
        buffer.reset_buffer()
        episode += batch_size

        lr_now = config.alg.lr * (
            1 - (episode * config.env.episode_length) / 20_000_000
        )
        agent.optimizer.param_groups[0]["lr"] = lr_now

        if logging:
            loss_dict.update({"reward": curr_reward})
            if episode % (50 * batch_size) == 0:
                agent.save_model(model_dir, episode * config.env.episode_length)
            wandb.log(
                loss_dict,
                step=episode * config.env.episode_length,
            )
