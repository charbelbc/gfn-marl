import torch
import numpy as np
from common.nets import (
    MPE_RNN_Actor,
    MPE_RNN_Critic,
    MPE_Actor,
    MPE_Critic,
)
from common.utils import MPE_ReplayBuffer
from common.config import Config
from alg.gflownet import EMGFlowNet
import os


class ValueNormalizer:
    def __init__(self, shape=(), epsilon=1e-5, beta=0.99):
        self.running_mean = torch.zeros(shape).float()
        self.running_mean_squared = torch.zeros(shape).float()
        self.debiasing_term = torch.tensor([0.0])
        self.epsilon = epsilon
        self.beta = beta

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_squared / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-4)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        batch_mean = values.mean(dim=(0, 1))
        batch_squared_mean = (values**2).mean(dim=(0, 1))

        self.running_mean.mul_(self.beta).add_(batch_mean * (1 - self.beta))
        self.running_mean_squared.mul_(self.beta).add_(
            batch_squared_mean * (1 - self.beta)
        )
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1 - self.beta))

    def normalize(self, values):
        mean, var = self.running_mean_var()
        mean = mean.view(1, 1, -1)
        var = var.view(1, 1, -1)
        return (values - mean) / (torch.sqrt(var))

    def denormalize(self, values):
        mean, var = self.running_mean_var()
        mean = mean.view(1, 1, -1)
        var = var.view(1, 1, -1)
        return values * torch.sqrt(var) + mean


class MPE_MAPPO:

    def __init__(
        self,
        config: Config,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):

        self.device = device
        self.n_agents = config.num_agents
        self.gamma = config.gamma
        self.gaelambda = config.gaelambda
        self.ppo_epochs = config.ppo_epochs
        self.eps_clip = config.eps_clip
        self.action_dim = config.action_dim
        self.use_rnn = config.use_rnn
        self.minibatch_size = config.minibatch_size
        self.normalize_value = config.normalize_value
        self.value_clipping = config.value_clipping

        if self.use_rnn:
            self.actor = MPE_RNN_Actor(config).to(self.device)
            self.critic = MPE_RNN_Critic(config).to(self.device)
            self.ac_parameters = list(self.actor.parameters()) + list(
                self.critic.parameters()
            )
        else:
            self.actor = MPE_Actor(
                action_dim=config.action_dim, obs_dim=config.obs_dim
            ).to(self.device)
            self.critic = MPE_Critic(state_dim=config.state_dim).to(self.device)
            self.ac_parameters = list(self.actor.parameters()) + list(
                self.critic.parameters()
            )
        self.optimizer = torch.optim.Adam(self.ac_parameters, lr=config.lr, eps=1e-5)
        if self.normalize_value:
            self.value_norm = ValueNormalizer(shape=self.n_agents)

    def select_action(self, obs, actor_memory=None, critic_memory=None):

        with torch.no_grad():
            obs = torch.from_numpy(np.stack(obs, axis=0)).float().to(self.device)
            if self.use_rnn:
                logits, actor_memory = self.actor(obs, actor_memory)
                value, critic_memory = self.critic(
                    obs.flatten(1).unsqueeze(1).repeat(1, self.n_agents, 1),
                    critic_memory,
                )
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprobs = dist.log_prob(action)

                return action, logprobs, value.squeeze(), actor_memory, critic_memory
            else:
                logits = self.actor(obs)
                value = self.critic(
                    obs.flatten(1).unsqueeze(1).repeat(1, self.n_agents, 1)
                )
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprobs = dist.log_prob(action)

                return action, logprobs, value.squeeze()

    def update(self, buffer: MPE_ReplayBuffer):

        rewards = torch.tensor(buffer.buffer["rewards"])
        values = torch.tensor(buffer.buffer["state_values"]).detach()
        if self.normalize_value:
            values = self.value_norm.denormalize(values)
        dones = torch.tensor(buffer.buffer["is_terminals"])
        batch, max_T, _ = rewards.shape

        with torch.no_grad():
            deltas = rewards + self.gamma * values[:, 1:] * (1 - dones) - values[:, :-1]
            gae = torch.zeros_like(deltas[:, 0])
            advantages = []
            for t in reversed(range(max_T)):
                gae = deltas[:, t] + self.gamma * self.gaelambda * gae * (
                    1 - dones[:, t]
                )
                advantages.insert(0, gae)
            advantages = torch.stack(advantages, dim=1)
            returns = advantages + values[:, :-1]

        if self.normalize_value:
            self.value_norm.update(returns)
            returns = self.value_norm.normalize(returns).to(self.device)
            old_values = self.value_norm.normalize(values[:, :-1]).to(self.device)
        else:
            returns = returns.to(self.device)
            old_values = values[:, :-1].to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = advantages.to(self.device)
        old_states = torch.tensor(buffer.buffer["states"]).to(self.device)
        old_actions = torch.tensor(buffer.buffer["actions"]).to(self.device)
        old_logprobs = torch.tensor(buffer.buffer["log_probs"]).detach().to(self.device)

        for _ in range(self.ppo_epochs):

            for index in torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(range(batch)),
                self.minibatch_size,
                False,
            ):

                if self.use_rnn:
                    actor_memory = torch.zeros(
                        self.minibatch_size, self.n_agents, 64
                    ).to(self.device)
                    critic_memory = torch.zeros(
                        self.minibatch_size, self.n_agents, 64
                    ).to(self.device)
                    logits_now, values_now = [], []
                    for t in range(max_T):
                        logits, actor_memory = self.actor(
                            old_states[index, t].float(), actor_memory
                        )
                        value, critic_memory = self.critic(
                            old_states[index, t]
                            .flatten(1)
                            .unsqueeze(1)
                            .repeat(1, self.n_agents, 1)
                            .float(),
                            critic_memory,
                        )
                        logits_now.append(logits)
                        values_now.append(value.squeeze(-1))
                    logits_now = torch.stack(logits_now, dim=1)
                    values_now = torch.stack(values_now, dim=1).squeeze(-1)
                else:
                    logits_now = self.actor(old_states[index].float())
                    values_now = self.critic(
                        old_states[index]
                        .flatten(2)
                        .unsqueeze(2)
                        .repeat(1, 1, self.n_agents, 1)
                        .float()
                    ).squeeze(-1)

                distribution_now = torch.distributions.Categorical(logits=logits_now)
                logprobs_now = distribution_now.log_prob(old_actions[index])
                entropy = distribution_now.entropy()
                ratios = torch.exp(logprobs_now - old_logprobs[index])

                surr1 = ratios * advantages[index]
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages[index]
                )
                policy_loss = -torch.min(surr1, surr2) - 0.01 * entropy

                if self.value_clipping:
                    value_clipped = old_values[index] + torch.clamp(
                        values_now - old_values[index], -self.eps_clip, self.eps_clip
                    )
                    value_surr1 = (values_now - returns[index]).pow(2)
                    value_surr2 = (value_clipped - returns[index]).pow(2)
                    value_loss = torch.max(value_surr1, value_surr2)
                else:
                    value_loss = (values_now - returns[index]).pow(2)
                    # value_loss = torch.nn.functional.smooth_l1_loss(
                    #     values_now, returns[index], reduction="none", beta=10.0
                    # )

                loss = policy_loss.mean() + 0.5 * value_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ac_parameters, max_norm=10.0)
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=10.0
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=10.0
                )
                self.optimizer.step()

        losses = {
            "actor_loss": policy_loss.mean().item(),
            "critic_loss": value_loss.mean().item(),
            "entropy": entropy.mean().item(),
            "ratios": ratios.mean().item(),
            "adv_abs_mean": advantages.abs().mean().item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }
        return losses

    def save_model(self, model_dir, step):

        save_dict = {
            "step": step,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        save_path = os.path.join(model_dir, "model_mf.pth")
        torch.save(save_dict, save_path)

    def load_model(self, saved_dict):

        self.actor.load_state_dict(saved_dict["actor"])
        self.critic.load_state_dict(saved_dict["critic"])


class MPE_GFN_MAPPO:

    def __init__(
        self,
        config: Config,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):

        self.device = device
        self.n_agents = config.num_agents
        self.gamma = config.gamma
        self.gaelambda = config.gaelambda
        self.ppo_epochs = config.ppo_epochs
        self.eps_clip = config.eps_clip
        self.action_dim = config.action_dim
        self.use_rnn = config.use_rnn
        self.minibatch_size = config.minibatch_size
        self.normalize_value = config.normalize_value
        self.value_clipping = config.value_clipping

        if self.use_rnn:
            self.actor = MPE_RNN_Actor(config).to(self.device)
            self.critic = MPE_RNN_Critic(config).to(self.device)
            self.ac_parameters = list(self.actor.parameters()) + list(
                self.critic.parameters()
            )
        else:
            self.actor = MPE_Actor(
                action_dim=config.action_dim, obs_dim=config.obs_dim
            ).to(self.device)
            self.critic = MPE_Critic(state_dim=config.state_dim).to(self.device)
            self.ac_parameters = list(self.actor.parameters()) + list(
                self.critic.parameters()
            )
        self.optimizer = torch.optim.Adam(self.ac_parameters, lr=config.lr, eps=1e-5)
        if self.normalize_value:
            self.value_norm = ValueNormalizer(shape=self.n_agents)

        self.gflownet = EMGFlowNet(device=device, config=config)
        self.gfn_sampling_exponent = config.gfn_sampling_exponent

    def select_action(
        self,
        obs,
        prev_actions,
        rewards,
        gfn_memory,
        actor_memory=None,
        critic_memory=None,
    ):

        with torch.no_grad():
            obs = torch.from_numpy(np.stack(obs, axis=0)).float().to(self.device)
            _, latents, next_gfn_memory = self.gflownet.sample_latents(
                obs,
                prev_actions,
                rewards,
                gfn_memory,
                rand_prob=0,
                prob_exponent=self.gfn_sampling_exponent,
            )
            if self.use_rnn:
                logits, actor_memory = self.actor(
                    obs, actor_memory, latents.float().to(self.device)
                )
                value, critic_memory = self.critic(
                    obs.flatten(1).unsqueeze(1).repeat(1, self.n_agents, 1),
                    critic_memory,
                    latents.unsqueeze(1)
                    .repeat_interleave(self.n_agents, 1)
                    .float()
                    .to(self.device),
                )
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprobs = dist.log_prob(action)

                return (
                    action,
                    logprobs,
                    value.squeeze(),
                    actor_memory,
                    critic_memory,
                    latents,
                    next_gfn_memory,
                )
            else:
                logits = self.actor(obs)
                value = self.critic(
                    obs.flatten(1).unsqueeze(1).repeat(1, self.n_agents, 1)
                )
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprobs = dist.log_prob(action)

                return action, logprobs, value.squeeze()

    def update(self, buffer: MPE_ReplayBuffer):

        rewards = torch.tensor(buffer.buffer["rewards"])
        values = torch.tensor(buffer.buffer["state_values"]).detach()
        if self.normalize_value:
            values = self.value_norm.denormalize(values)
        dones = torch.tensor(buffer.buffer["is_terminals"])
        batch, max_T, _ = rewards.shape

        with torch.no_grad():
            deltas = rewards + self.gamma * values[:, 1:] * (1 - dones) - values[:, :-1]
            gae = torch.zeros_like(deltas[:, 0])
            advantages = []
            for t in reversed(range(max_T)):
                gae = deltas[:, t] + self.gamma * self.gaelambda * gae * (
                    1 - dones[:, t]
                )
                advantages.insert(0, gae)
            advantages = torch.stack(advantages, dim=1)
            returns = advantages + values[:, :-1]

        if self.normalize_value:
            self.value_norm.update(returns)
            returns = self.value_norm.normalize(returns).to(self.device)
            old_values = self.value_norm.normalize(values[:, :-1]).to(self.device)
        else:
            returns = returns.to(self.device)
            old_values = values[:, :-1].to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = advantages.to(self.device)
        old_states = torch.tensor(buffer.buffer["states"]).to(self.device)
        old_actions = torch.tensor(buffer.buffer["actions"]).to(self.device)
        old_logprobs = torch.tensor(buffer.buffer["log_probs"]).detach().to(self.device)
        latents = torch.tensor(buffer.buffer["latents"]).detach().to(self.device)

        for _ in range(self.ppo_epochs):

            for index in torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(range(batch)),
                self.minibatch_size,
                False,
            ):

                if self.use_rnn:
                    actor_memory = torch.zeros(
                        self.minibatch_size, self.n_agents, 64
                    ).to(self.device)
                    critic_memory = torch.zeros(
                        self.minibatch_size, self.n_agents, 64
                    ).to(self.device)
                    logits_now, values_now = [], []
                    for t in range(max_T):
                        logits, actor_memory = self.actor(
                            old_states[index, t].float(),
                            actor_memory,
                            latents[index, t].float(),
                        )
                        value, critic_memory = self.critic(
                            old_states[index, t]
                            .flatten(1)
                            .unsqueeze(1)
                            .repeat(1, self.n_agents, 1)
                            .float(),
                            critic_memory,
                            latents[index, t]
                            .unsqueeze(1)
                            .repeat_interleave(self.n_agents, 1)
                            .float(),
                        )
                        logits_now.append(logits)
                        values_now.append(value.squeeze(-1))
                    logits_now = torch.stack(logits_now, dim=1)
                    values_now = torch.stack(values_now, dim=1).squeeze(-1)
                else:
                    logits_now = self.actor(old_states[index].float())
                    values_now = self.critic(
                        old_states[index]
                        .flatten(2)
                        .unsqueeze(2)
                        .repeat(1, 1, self.n_agents, 1)
                        .float()
                    ).squeeze(-1)

                distribution_now = torch.distributions.Categorical(logits=logits_now)
                logprobs_now = distribution_now.log_prob(old_actions[index])
                entropy = distribution_now.entropy()
                ratios = torch.exp(logprobs_now - old_logprobs[index])

                surr1 = ratios * advantages[index]
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages[index]
                )
                policy_loss = -torch.min(surr1, surr2) - 0.01 * entropy

                if self.value_clipping:
                    value_clipped = old_values[index] + torch.clamp(
                        values_now - old_values[index], -self.eps_clip, self.eps_clip
                    )
                    value_surr1 = (values_now - returns[index]).pow(2)
                    value_surr2 = (value_clipped - returns[index]).pow(2)
                    value_loss = torch.max(value_surr1, value_surr2)
                else:
                    value_loss = (values_now - returns[index]).pow(2)
                    # value_loss = torch.nn.functional.smooth_l1_loss(
                    #     values_now, returns[index], reduction="none", beta=10.0
                    # )

                loss = policy_loss.mean() + 0.5 * value_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ac_parameters, max_norm=10.0)
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=10.0
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=10.0
                )
                self.optimizer.step()

        losses = {
            "actor_loss": policy_loss.mean().item(),
            "critic_loss": value_loss.mean().item(),
            "entropy": entropy.mean().item(),
            "ratios": ratios.mean().item(),
            "adv_abs_mean": advantages.abs().mean().item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }
        return losses

    def save_model(self, model_dir, step):

        save_dict = {
            "step": step,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "gflownet": self.gflownet.state_dict(),
        }
        save_path = os.path.join(model_dir, "model_gfn.pth")
        torch.save(save_dict, save_path)

    def load_model(self, saved_dict):

        self.actor.load_state_dict(saved_dict["actor"])
        self.critic.load_state_dict(saved_dict["critic"])
        self.gflownet.load_state_dict(saved_dict["gflownet"])
