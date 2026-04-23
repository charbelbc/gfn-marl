import torch
import numpy as np
from common.config import Config
import gymnasium as gym
from alg.mappo import MAPPO, MPE_MAPPO
from common.utils import ReplayBuffer, ParallelEnv, MPE_ReplayBuffer
from mpe.MPE_env import MPEEnv

import multigrid.envs
import gymnasium as gym
import minigrid
from multigrid.wrappers import FullyObsWrapper

import wandb


def train(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    batch_size = config.batch_size
    agent = MAPPO(
        device=device,
        n_agents=config.n_agents,
        gamma=config.gamma,
        gaelambda=config.gaelambda,
        ppo_epochs=config.ppo_epochs,
        eps_clip=config.eps_clip,
        lr=config.lr,
        memory_size=config.memory_size,
        action_dim=config.action_dim,
    )
    buffer = ReplayBuffer(
        batch_size=batch_size, ep_limit=config.max_steps, n_agents=config.n_agents
    )
    envs = [
        FullyObsWrapper(
            gym.make(
                config.env_name,
                agents=config.n_agents,
                render_mode="rgb_array",
                max_steps=config.max_steps,
            )
        )
        for _ in range(config.batch_size)
    ]
    env = ParallelEnv(envs)

    episode = 0

    while episode < 4_000_000:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        doness = dones.clone()
        memory = torch.zeros(batch_size, config.n_agents, 2 * config.memory_size).to(
            agent.device
        )
        buffer.buffer["instructions"] = [o[0][0]["mission"].string for o in obs]
        step = 0
        curr_reward = 0.0
        success_rate = 0.0

        while not doness.all():
            actions, memory, logits, value = agent.select_action(obs, memory)
            next_obs = env.step(
                list(
                    {j: act[j].item() for j in range(agent.n_agents)} for act in actions
                )
            )
            rewards = torch.tensor([o[1][0] for o in next_obs])
            dones = torch.logical_or(
                torch.stack([torch.tensor(o[2][0]) for o in next_obs]),
                torch.stack([torch.tensor(o[3][0]) for o in next_obs]),
            )
            buffer.store_transition(
                step,
                obs,
                actions.squeeze().cpu(),
                torch.gather(logits, -1, actions.unsqueeze(-1)).squeeze().cpu(),
                value.squeeze().cpu(),
                rewards,
                dones,
            )
            for e in range(batch_size):
                if dones[e] and not doness[e]:
                    doness[e] = True
                    curr_reward += rewards[e]
                    success_rate += rewards[e] > 0
                    buffer.buffer["lengths"][e] = step
            step += 1
            obs = next_obs

        agent.update(buffer)
        buffer.reset_buffer()
        episode += batch_size

        print(episode)
        if logging:
            wandb.log(
                {
                    "reward": curr_reward / batch_size,
                    "succes_rate": success_rate / batch_size,
                },
                step=episode,
            )
            if episode % 100_000 == 0:
                torch.save(agent.policy.state_dict(), "model")


def train_mpe(
    config: Config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    logging: bool = True,
):

    batch_size = config.batch_size
    agent = MPE_MAPPO(
        device=device,
        n_agents=config.num_agents,
        gamma=config.gamma,
        gaelambda=config.gaelambda,
        ppo_epochs=config.ppo_epochs,
        eps_clip=config.eps_clip,
        lr=config.lr,
        action_dim=config.action_dim,
    )
    buffer = MPE_ReplayBuffer(
        batch_size=batch_size,
        ep_limit=config.episode_length,
        n_agents=config.num_agents,
    )
    envs = [MPEEnv(config) for _ in range(batch_size)]
    env = ParallelEnv(envs)

    episode = 0

    while episode < 4_000_000:

        obs = env.reset()
        dones = torch.zeros(batch_size, dtype=bool)
        doness = dones.clone()
        step = 0
        curr_reward = 0.0

        while not doness.all():
            actions, logits, value = agent.select_action(obs)
            next_obs = env.step(actions.cpu())
            rewards = torch.tensor([o[1][0] for o in next_obs])
            dones = torch.tensor([o[2][0] for o in next_obs])
            buffer.store_transition(
                step,
                obs,
                actions.argmax(-1).cpu(),
                torch.gather(logits, -1, actions.argmax(-1).unsqueeze(-1))
                .squeeze()
                .cpu(),
                value.squeeze().cpu(),
                rewards.squeeze(),
                dones,
            )
            curr_reward += rewards.mean().item()
            obs = [o[0] for o in next_obs]
            for e in range(batch_size):
                if dones[e] and not doness[e]:
                    doness[e] = True
                    buffer.buffer["lengths"][e] = step
                    _, _, value = agent.select_action(obs)
                    buffer.buffer["state_values"][e] = value[e]
            step += 1

        agent.update(buffer)
        buffer.reset_buffer()
        episode += batch_size

        # print(episode)
        if logging:
            wandb.log(
                {
                    "reward": curr_reward / batch_size,
                },
                step=episode * config.episode_length,
            )
            # print(episode, curr_reward)
            if episode % 100_000 == 0:
                torch.save(agent.policy.state_dict(), "model")
