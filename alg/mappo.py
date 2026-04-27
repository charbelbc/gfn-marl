import torch
import numpy as np
from common.nets import (
    ACNetwork,
    InstructionsPreprocessor,
    MPE_ACNetwork,
    MPE_RNN_Actor,
    MPE_RNN_Critic,
    MPE_Actor,
    MPE_Critic,
)
from common.utils import ReplayBuffer


class MAPPO:

    def __init__(
        self,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_agents: int = 2,
        gamma: float = 0.99,
        gaelambda: float = 0.95,
        ppo_epochs: int = 10,
        eps_clip: float = 0.2,
        lr: float = 0.001,
        memory_size: int = 512,
        action_dim: int = 5,
    ):

        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.gaelambda = gaelambda
        self.ppo_epochs = ppo_epochs
        self.eps_clip = eps_clip
        self.memory_size = memory_size
        self.action_dim = action_dim

        self.policy = ACNetwork(
            n_agents=n_agents, memory_size=memory_size, action_dim=action_dim
        ).to(self.device)
        self.policy.train()
        self.instr_preprocessor = InstructionsPreprocessor()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs, memory):

        with torch.no_grad():
            logits, value, next_memory = self.policy(
                torch.stack(
                    [
                        torch.stack(
                            [
                                torch.from_numpy(o[0][i]["image"]).permute(2, 0, 1)
                                for i in range(self.n_agents)
                            ]
                        )
                        for o in obs
                    ]
                ).to(self.device),
                memory,
                self.instr_preprocessor([o[0][0]["mission"].string for o in obs]).to(
                    self.device
                ),
            )
            action = (
                logits.flatten(0, 1)
                .softmax(-1)
                .multinomial(1)
                .reshape(memory.shape[0], self.n_agents)
            )

        return action, next_memory, logits, value

    def update(self, buffer: ReplayBuffer):

        rewards = torch.tensor(buffer.buffer["rewards"])
        values = torch.tensor(buffer.buffer["state_values"])
        dones = torch.tensor(buffer.buffer["is_terminals"])
        lengths = torch.tensor(buffer.buffer["lengths"]).int() + 1
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        batch, max_T = rewards.shape
        for e in range(batch):
            done_idx = torch.where(dones[e])[0]
            if len(done_idx) > 0:
                L = done_idx[0].item() + 1
            else:
                L = max_T
            r = rewards[e, :L]
            v = torch.cat([values[e, :L], torch.tensor([0])])
            d = dones[e, :L]
            gae = 0
            for t in reversed(range(L)):
                delta = r[t] + 0.99 * v[t + 1] * (1 - d[t]) - v[t]
                gae = delta + 0.99 * 0.99 * (1 - d[t]) * gae
                advantages[e, t] = gae
            returns[e, :L] = advantages[e, :L] + values[e, :L]
        advantages = torch.cat([advantages[e, : lengths[e]] for e in range(batch)]).to(
            self.device
        )
        returns = torch.cat([returns[e, : lengths[e]] for e in range(batch)]).to(
            self.device
        )
        old_values = torch.cat([values[e, : lengths[e]] for e in range(batch)]).to(
            self.device
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.tensor(buffer.buffer["states"]).to(self.device)
        old_actions = torch.cat(
            [
                torch.tensor(buffer.buffer["actions"][e, : lengths[e]])
                for e in range(batch)
            ]
        ).to(self.device)
        old_logprobs = torch.cat(
            [
                torch.tensor(buffer.buffer["log_probs"][e, : lengths[e]])
                for e in range(batch)
            ]
        ).to(self.device)

        for _ in range(self.ppo_epochs):

            memory = torch.zeros(
                old_states.shape[0], self.n_agents, 2 * self.memory_size
            ).to(self.device)
            instr = self.instr_preprocessor(buffer.buffer["instructions"]).to(
                self.device
            )

            logits_now, values_now = [], []

            for t in range(old_states.shape[1]):

                logits, values, memory = self.policy(
                    old_states[
                        :,
                        t,
                    ],
                    memory,
                    instr,
                )

                logits_now.append(logits)
                values_now.append(values)

            logits_now = torch.stack(logits_now, dim=1)
            logits_now = torch.cat(
                [logits_now[i, : lengths[i]] for i in range(len(lengths))],
                dim=0,
            )

            values_now = torch.stack(values_now, dim=1).squeeze(-1)
            values_now = torch.cat(
                [values_now[i, : lengths[i]] for i in range(len(lengths))],
                dim=0,
            )

            distribution_now = torch.distributions.Categorical(logits=logits_now)
            logprobs_now = distribution_now.log_prob(old_actions)
            entropy = distribution_now.entropy()
            ratios = torch.exp(logprobs_now - old_logprobs)

            surr1 = ratios * advantages.unsqueeze(1).repeat(1, self.n_agents)
            surr2 = torch.clamp(
                ratios, 1 - self.eps_clip, 1 + self.eps_clip
            ) * advantages.unsqueeze(1).repeat(1, self.n_agents)
            policy_loss = -torch.min(surr1, surr2).mean()

            value_clipped = old_values + torch.clamp(
                values_now - old_values, -self.eps_clip, self.eps_clip
            )
            value_surr1 = (values_now - returns).pow(2)
            value_surr2 = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_surr1, value_surr2).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()


class MPE_MAPPO:

    def __init__(
        self,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_agents: int = 2,
        gamma: float = 0.99,
        gaelambda: float = 0.95,
        ppo_epochs: int = 10,
        eps_clip: float = 0.2,
        lr: float = 0.001,
        action_dim: int = 5,
        use_rnn: bool = False,
        minibatch_size: int = 8,
    ):

        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.gaelambda = gaelambda
        self.ppo_epochs = ppo_epochs
        self.eps_clip = eps_clip
        self.action_dim = action_dim
        self.use_rnn = use_rnn
        self.minibatch_size = minibatch_size

        if self.use_rnn:
            self.actor = MPE_RNN_Actor(action_dim=action_dim, n_agents=n_agents).to(
                self.device
            )
            self.critic = MPE_RNN_Critic(action_dim=action_dim, n_agents=n_agents).to(
                self.device
            )
            self.ac_parameters = list(self.actor.parameters()) + list(
                self.critic.parameters()
            )
        else:
            # self.policy = MPE_ACNetwork(action_dim=action_dim, n_agents=n_agents).to(
            #     self.device
            # )
            self.actor = MPE_Actor(action_dim=action_dim, n_agents=n_agents).to(
                self.device
            )
            self.critic = MPE_Critic(action_dim=action_dim, n_agents=n_agents).to(
                self.device
            )
            self.ac_parameters = list(self.actor.parameters()) + list(
                self.critic.parameters()
            )
        # self.policy.train()
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.optimizer = torch.optim.Adam(self.ac_parameters, lr=lr, eps=1e-5)

    def select_action(self, obs):

        with torch.no_grad():
            obs = torch.from_numpy(np.stack(obs)).float().to(self.device)
            # logits, value = self.policy(obs)
            logits = self.actor(obs)
            value = self.critic(obs.flatten(1).unsqueeze(1).repeat(1, self.n_agents, 1))
            action = torch.nn.functional.one_hot(
                logits.softmax(-1)
                .flatten(0, 1)
                .multinomial(1)
                .reshape(obs.shape[0], obs.shape[1]),
                5,
            )

        return action, logits, value.squeeze()

    def update(self, buffer: ReplayBuffer):

        rewards = torch.tensor(buffer.buffer["rewards"])
        values = torch.tensor(buffer.buffer["state_values"]).detach()
        dones = torch.tensor(buffer.buffer["is_terminals"])
        batch, max_T, _ = rewards.shape

        advantages = []
        gae = 0
        with torch.no_grad():
            deltas = rewards + self.gamma * values[:, 1:] * (1 - dones) - values[:, :-1]
            for t in reversed(range(max_T)):
                gae = deltas[:, t] + self.gamma * self.gaelambda * gae
                advantages.insert(0, gae)
            advantages = torch.stack(advantages, dim=1)
            returns = advantages + values[:, :-1]
        # advantages = torch.zeros_like(rewards)
        # returns = torch.zeros_like(rewards)
        # batch, max_T, _ = rewards.shape
        # for e in range(batch):
        #     done_idx = torch.where(dones[e])[0]
        #     if len(done_idx) > 0:
        #         L = done_idx[0].item() + 1
        #     else:
        #         L = max_T
        #     r = rewards[e, :L]
        #     v = values[e, : L + 1]
        #     d = dones[e, :L]
        #     gae = 0
        #     for t in reversed(range(L)):
        #         delta = r[t] + self.gamma * v[t + 1] * (1 - d[t]) - v[t]
        #         gae = delta + self.gaelambda * self.gamma * (1 - d[t]) * gae
        #         advantages[e, t] = gae
        #     returns[e, :L] = advantages[e, :L] + values[e, :L]
        # advantages = torch.cat([advantages[e, : lengths[e]] for e in range(batch)]).to(
        #     self.device
        # )
        # returns = torch.cat([returns[e, : lengths[e]] for e in range(batch)]).to(
        #     self.device
        # )
        # old_values = (
        #     torch.cat([values[e, : lengths[e]] for e in range(batch)])
        #     .to(self.device)
        #     .detach()
        # )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_values = values[:, :-1].to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.tensor(buffer.buffer["states"]).to(self.device)
        # old_actions = torch.cat(
        #     [
        #         torch.tensor(buffer.buffer["actions"][e, : lengths[e]])
        #         for e in range(batch)
        #     ]
        # ).to(self.device)
        old_actions = torch.tensor(buffer.buffer["actions"]).to(self.device)
        # old_logprobs = (
        #     torch.cat(
        #         [
        #             torch.tensor(buffer.buffer["log_probs"][e, : lengths[e]])
        #             for e in range(batch)
        #         ]
        #     )
        #     .detach()
        #     .to(self.device)
        # )
        old_logprobs = torch.tensor(buffer.buffer["log_probs"]).detach().to(self.device)

        for _ in range(self.ppo_epochs):

            for index in torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(range(batch)),
                self.minibatch_size,
                False,
            ):

                if self.use_rnn:
                    self.actor.actor_rnn_hidden = None
                    self.critic.critic_rnn_hidden = None
                    logits_now, values_now = [], []
                    for t in range(max_T):
                        logits = self.actor(old_states[index, t].float())
                        value = self.critic(
                            old_states[index, t]
                            .flatten(1)
                            .unsqueeze(1)
                            .repeat(1, self.n_agents, 1)
                            .float()
                        ).squeeze(-1)
                        logits_now.append(logits)
                        values_now.append(value)
                    logits_now = torch.stack(logits_now, dim=1)
                    values_now = torch.stack(values_now, dim=1).squeeze(-1)
                else:
                    # logits_now, values_now = self.policy(old_states.flatten(0, 1).float())
                    logits_now = self.actor(old_states[index].float())
                    values_now = self.critic(
                        old_states[index]
                        .flatten(2)
                        .unsqueeze(2)
                        .repeat(1, 1, self.n_agents, 1)
                        .float()
                    ).squeeze(-1)
                    # logits_now = logits_now.reshape(batch, max_T, self.n_agents, -1)
                    # values_now = values_now.reshape(batch, max_T)
                # logits_now = torch.cat(
                #     [logits_now[i, : lengths[i]] for i in range(len(lengths))],
                #     dim=0,
                # )
                # values_now = torch.cat(
                #     [values_now[i, : lengths[i]] for i in range(len(lengths))],
                #     dim=0,
                # )

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

                # value_clipped = old_values + torch.clamp(
                #     values_now - old_values, -self.eps_clip, self.eps_clip
                # )
                # value_surr1 = (values_now - returns).pow(2)
                # value_surr2 = (value_clipped - returns).pow(2)
                # value_loss = torch.max(value_surr1, value_surr2).mean()
                # value_loss = (values_now - returns[index]).pow(2)
                value_loss = torch.nn.functional.smooth_l1_loss(
                    values_now, returns[index], reduction="none"
                )

                loss = policy_loss.mean() + 0.5 * value_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
                torch.nn.utils.clip_grad_norm_(self.ac_parameters, max_norm=10.0)
                self.optimizer.step()

        losses = {
            "actor_loss": policy_loss.mean().item(),
            "critic_loss": value_loss.mean().item(),
            "entropy": entropy.mean().item(),
        }
        return losses
