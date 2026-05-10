from multiprocessing import Process, Pipe
import gymnasium as gym
import numpy as np
import torch
import random
from common.config import Config
from multiagent.make_env import make_env


def my_f(config):
    return make_env("simple_cn", config)


def worker(remote, parent_remote, env_fn, config, seed):
    parent_remote.close()
    np.random.seed(seed)
    random.seed(seed)
    env = env_fn(config)
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
    def __init__(self, env_fn, config, n_envs, base_seed=42):
        self.n_envs = n_envs
        env = env_fn(config)
        self.obs_dim = env.observation_space[0].shape[0]
        env.close()

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])

        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            seed = base_seed + i
            p = Process(target=worker, args=(work_remote, remote, env_fn, config, seed))
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


class MPE_ReplayBuffer:
    def __init__(
        self,
        config: Config,
    ):
        self.batch_size = config.batch_size
        self.ep_limit = config.episode_length
        self.n_agents = config.num_agents
        self.obs_dim = config.obs_dim
        self.gfn_state_size = config.gfn_state_size
        # if config.use_gfn:
        # self.obs_dim += config.gfn_state_size * (config.num_agents - 1)
        self.buffer = None
        self.reset_buffer()

        self.episode = 0

    def reset_buffer(self):
        self.buffer = {
            "states": np.zeros(
                [self.batch_size, self.ep_limit, self.n_agents, self.obs_dim]
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
            "latents": np.zeros(
                [
                    self.batch_size,
                    self.ep_limit,
                    self.n_agents,
                    self.n_agents - 1,
                    self.gfn_state_size,
                ],
                dtype=float,
            ),
        }

    def store_transition(
        self, step, obs, actions, log_probs, state_values, rewards, dones, latents=None
    ):

        obs = np.stack(obs, axis=0)
        self.buffer["states"][:, step] = obs
        self.buffer["actions"][:, step] = actions
        self.buffer["log_probs"][:, step] = log_probs
        self.buffer["rewards"][:, step] = rewards
        self.buffer["state_values"][:, step] = state_values
        self.buffer["is_terminals"][:, step] = dones
        if latents is not None:
            self.buffer["latents"][:, step] = latents.numpy()


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
