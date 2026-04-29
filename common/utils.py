from multiprocessing import Process, Pipe
import gymnasium as gym
import numpy as np
import torch

from mpe.scenarios.simple_spread import Scenario
import mpe.environment
from multiagent.scenarios.simple_spread import Scenario
import multiagent.environment


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            # obs, reward, terminated, truncated, info = env.step(data)
            # if terminated[0] or truncated[0]:
            #     obs, _ = env.reset()
            # conn.send((obs, reward, terminated, truncated, info))
            obs, reward, terminated, info = env.step(data)
            conn.send((obs, reward, terminated, info))
        elif cmd == "reset":
            # obs, _ = env.reset()
            # conn.send((obs,))
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        # results = [(self.envs[0].reset()[0],)] + [local.recv() for local in self.locals]
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        # obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        # if terminated[0] or truncated[0]:
        #     obs, _ = self.envs[0].reset()
        # results = [(obs, reward, terminated, truncated, info)] + [
        #     local.recv() for local in self.locals
        # ]
        obs, reward, terminated, info = self.envs[0].step(actions[0])
        results = [(obs, reward, terminated, info)] + [
            local.recv() for local in self.locals
        ]
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()


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
            "actions": np.empty(
                [self.batch_size, self.ep_limit, self.n_agents], dtype=int
            ),
            "log_probs": np.empty([self.batch_size, self.ep_limit, self.n_agents]),
            "rewards": np.empty([self.batch_size, self.ep_limit, self.n_agents]),
            "state_values": np.empty(
                [self.batch_size, self.ep_limit + 1, self.n_agents]
            ),
            "is_terminals": np.empty([self.batch_size, self.ep_limit, self.n_agents]),
            "lengths": np.empty([self.batch_size]),
        }

    def store_transition(
        self, step, obs, actions, log_probs, state_values, rewards, dones
    ):
        self.buffer["states"][:, step] = torch.from_numpy(np.stack(obs)).float()
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
