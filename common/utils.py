from multiprocessing import Process, Pipe
import gymnasium as gym
import numpy as np
import torch
import random

from mpe.scenarios.simple_spread import Scenario
import mpe.environment
from multiagent.scenarios.simple_spread import Scenario
import multiagent.environment
from multiagent.make_env import make_env


def my_f():
    return make_env("simple_spread")


def worker(remote, parent_remote, env_fn, seed):
    parent_remote.close()
    np.random.seed(seed)
    random.seed(seed)
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            # if done:
            # obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            remote.send(obs)
        else:
            raise NotImplementedError


class ParallelEnv:
    def __init__(self, env_fn, n_envs, base_seed=42):
        self.n_envs = n_envs

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])

        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            seed = base_seed + i
            p = Process(target=worker, args=(work_remote, remote, env_fn, seed))
            p.daemon = True
            p.start()
            work_remote.close()
            self.processes.append(p)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]

        # obs, rewards, dones, infos = zip(*results)
        # return list(obs), list(rewards), list(dones), list(infos)
        return results


class ReplayBuffer:
    def __init__(self, batch_size: int = 64, ep_limit: int = 50, n_agents: int = 2):
        self.batch_size = batch_size
        self.ep_limit = ep_limit
        self.n_agents = n_agents
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            "states": np.zeros(
                [self.batch_size, self.ep_limit, self.n_agents, 3, 8, 8]
            ),
            "actions": np.empty([self.batch_size, self.ep_limit, self.n_agents]),
            "log_probs": np.empty([self.batch_size, self.ep_limit, self.n_agents]),
            "rewards": np.empty([self.batch_size, self.ep_limit]),
            "state_values": np.empty([self.batch_size, self.ep_limit]),
            "is_terminals": np.empty([self.batch_size, self.ep_limit]),
            "lengths": np.empty([self.batch_size]),
            "instructions": np.empty([self.batch_size], dtype="<U100"),
        }

    def store_transition(
        self, step, obs, actions, log_probs, state_values, rewards, dones
    ):
        self.buffer["states"][:, step] = torch.stack(
            [
                torch.stack(
                    [
                        torch.from_numpy(o[0][i]["image"]).permute(2, 0, 1)
                        for i in range(self.n_agents)
                    ]
                )
                for o in obs
            ]
        )
        self.buffer["actions"][:, step] = actions
        self.buffer["log_probs"][:, step] = log_probs
        self.buffer["rewards"][:, step] = rewards
        self.buffer["state_values"][:, step] = state_values
        self.buffer["is_terminals"][:, step] = dones


class MPE_ReplayBuffer:
    def __init__(self, batch_size: int = 64, ep_limit: int = 50, n_agents: int = 2):
        self.batch_size = batch_size
        self.ep_limit = ep_limit
        self.n_agents = n_agents
        self.buffer = None
        self.reset_buffer()

        self.episode = 0

    def reset_buffer(self):
        self.buffer = {
            "states": np.zeros(
                [self.batch_size, self.ep_limit, self.n_agents, 6 * self.n_agents]
            ),
            "actions": np.zeros(
                [self.batch_size, self.ep_limit, self.n_agents], dtype=int
            ),
            "log_probs": np.zeros([self.batch_size, self.ep_limit, self.n_agents]),
            "rewards": np.zeros([self.batch_size, self.ep_limit, self.n_agents]),
            "state_values": np.zeros(
                [self.batch_size, self.ep_limit + 1, self.n_agents]
            ),
            "is_terminals": np.zeros([self.batch_size, self.ep_limit, self.n_agents]),
        }

    def store_transition(
        self, step, obs, actions, log_probs, state_values, rewards, dones
    ):
        self.buffer["states"][:, step] = np.stack(obs, axis=0)
        self.buffer["actions"][:, step] = actions
        self.buffer["log_probs"][:, step] = log_probs
        self.buffer["rewards"][:, step] = rewards
        self.buffer["state_values"][:, step] = state_values
        self.buffer["is_terminals"][:, step] = dones

    def store_transitionn(
        self,
        step,
        obs,
        actions,
        log_probs,
        state_values,
        rewards,
        dones,
    ):
        self.buffer["states"][self.episode, step] = torch.from_numpy(
            np.stack(obs)
        ).float()
        self.buffer["actions"][self.episode, step] = actions
        self.buffer["log_probs"][self.episode, step] = log_probs
        self.buffer["rewards"][self.episode, step] = rewards
        self.buffer["state_values"][self.episode, step] = state_values
        self.buffer["is_terminals"][self.episode, step] = dones


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0  # small value to avoid division issues

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class Normalization:
    def __init__(self, shape=()):
        self.running_ms = RunningMeanStd(shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        return (x - self.running_ms.mean) / self.running_ms.std
