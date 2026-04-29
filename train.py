import torch
import numpy as np
from common.config import Config
import gymnasium as gym
from alg.mappo import MAPPO, MPE_MAPPO
from common.utils import ReplayBuffer, ParallelEnv, MPE_ReplayBuffer
from mpe.MPE_env import MPEEnv
from multiagent.make_env import make_env
from common.utils import Normalization

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
        use_rnn=config.use_rnn,
        minibatch_size=config.minibatch_size,
        normalize_value=config.normalize_value,
        value_clipping=config.value_clipping,
    )
    buffer = MPE_ReplayBuffer(
        batch_size=batch_size,
        ep_limit=config.episode_length,
        n_agents=config.num_agents,
    )
    # envs = [MPEEnv(config) for _ in range(batch_size)]
    envs = [make_env("simple_spread") for _ in range(batch_size)]
    env = ParallelEnv(envs)

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
            actor_memory = torch.zeros(batch_size * config.num_agents, 64).to(device)
            critic_memory = torch.zeros(batch_size * config.num_agents, 64).to(device)

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
                # test_reward = test_mpe(agent, env, config)
                # loss_dict.update({"test_reward": test_reward})
                torch.save(agent.actor.state_dict(), "model")
            wandb.log(
                loss_dict,
                step=episode * config.episode_length,
            )


def test_mpe(agent, env, config):

    obs = env.reset()
    batch_size = config.batch_size
    dones = torch.zeros(batch_size, dtype=bool)
    doness = dones.clone()
    step = 0
    curr_reward = 0.0
    if config.use_rnn:
        agent.actor.actor_rnn_hidden = None
        agent.critic.critic_rnn_hidden = None

    while not doness.all():
        _, logits, _ = agent.select_action(obs)
        actions = torch.nn.functional.one_hot(
            logits.softmax(-1)
            .argmax(-1)
            .unsqueeze(-1)
            .flatten(0, 1)
            .reshape(batch_size, agent.n_agents),
            5,
        )
        next_obs = env.step(actions.cpu())
        rewards = torch.stack([torch.tensor(o[1]).squeeze() for o in next_obs])
        dones = torch.stack([torch.tensor(o[2]).squeeze() for o in next_obs])

        curr_reward += rewards.mean().item()
        obs = [o[0] for o in next_obs]
        for e in range(batch_size):
            if dones[e][0] and not doness[e]:
                doness[e] = True
        step += 1
    return curr_reward


def train_mpe_single(
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
        use_rnn=config.use_rnn,
        minibatch_size=config.minibatch_size,
        normalize_value=config.normalize_value,
        value_clipping=config.value_clipping,
    )
    buffer = MPE_ReplayBuffer(
        batch_size=batch_size,
        ep_limit=config.episode_length,
        n_agents=config.num_agents,
    )
    # envs = [MPEEnv(config) for _ in range(batch_size)]
    # envs = [make_env("simple_spread") for _ in range(batch_size)]
    # env = ParallelEnv(envs)
    env = make_env("simple_spread")

    episode = 0
    if config.reward_normalization:
        reward_norm = Normalization(config.num_agents)

    while (episode * config.episode_length) < 5_000_000:

        curr_reward = 0.0

        for _ in range(config.batch_size):

            obs = env.reset()
            done = False
            step = 0
            if config.use_rnn:
                actor_memory = torch.zeros(1 * config.num_agents, 64).to(device)
                critic_memory = torch.zeros(1 * config.num_agents, 64).to(device)

            while not done:
                if config.use_rnn:
                    actions, logits, value, actor_memory, critic_memory = (
                        agent.select_action([obs], actor_memory, critic_memory)
                    )
                else:
                    actions, logits, value = agent.select_action([obs])
                next_obs = env.step(actions.squeeze().cpu())
                rewards = torch.tensor(next_obs[1])
                if config.reward_normalization:
                    normalized_rewards = reward_norm(rewards)
                dones = torch.tensor(next_obs[2])
                buffer.store_transitionn(
                    step,
                    [obs],
                    actions.squeeze().cpu(),
                    logits.squeeze().cpu(),
                    value.squeeze().cpu(),
                    (
                        normalized_rewards.squeeze()
                        if config.reward_normalization
                        else rewards.squeeze()
                    ),
                    dones,
                )
                curr_reward += rewards[0].item()
                obs = next_obs[0]
                done = dones[0]
                if done:
                    _, _, value = agent.select_action([obs])
                    buffer.buffer["state_values"][buffer.episode, -1] = value.cpu()
                step += 1
            buffer.episode += 1

        loss_dict = agent.update(buffer)
        buffer.reset_buffer()
        buffer.episode = 0
        episode += batch_size

        lr_now = config.lr * (1 - (episode * config.episode_length) / 20_000_000)
        agent.optimizer.param_groups[0]["lr"] = lr_now
        # print(episode, curr_reward)

        if logging:
            loss_dict.update(
                {
                    "reward": curr_reward / batch_size,
                }
            )
            if episode % (50 * batch_size) == 0:
                # test_reward = test_mpe(agent, env, config)
                # loss_dict.update({"test_reward": test_reward})
                torch.save(agent.actor.state_dict(), "model")
            wandb.log(
                loss_dict,
                step=episode * config.episode_length,
            )
