import torch
from common.nets import MLP
from common.utils import MPE_ReplayBuffer
from common.config import Config
from common.nets import MPE_Actor, MPE_RNN_Actor


class SupTom_RNNActor(MPE_RNN_Actor):

    def __init__(
        self,
        config: Config,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(config=config)

        self.device = device
        self.obs_dim = config.env.obs_dim
        self.num_agents = config.env.num_agents
        self.goal_size = config.module.goal_size

        self.belief_net = MLP(
            input_size=config.training.actor.memory_size,
            output_size=sum(config.module.goal_size),
        )

        self.others_belief_net = MLP(
            input_size=config.training.actor.memory_size,
            output_size=(config.env.num_agents - 1) * sum(config.module.goal_size),
        )

        self.belief_net_optimizer = torch.optim.Adam(
            self.belief_net.parameters(), lr=config.module.belief_lr
        )

    def forward(self, observations, actor_rnn_hidden):

        batch, agents, features = observations.shape
        x = self.fc(observations.flatten(0, 1))
        actor_memory = self.rnn(self.mem_norm(x), actor_rnn_hidden.flatten(0, 1)).view(
            batch, agents, -1
        )

        # Agent own beliefs, shape: [batch, agents, sum(goal_size)]
        beliefs = self.belief_net(actor_memory)

        # Computing a separate softmax for each goal
        chunked_beliefs = beliefs.split(self.goal_size, dim=-1)
        beliefs = torch.cat(
            [torch.nn.functional.softmax(chunk, dim=-1) for chunk in chunked_beliefs],
            dim=-1,
        )

        others_beliefs = self.others_belief_net(actor_memory).view(
            batch, agents, agents - 1, -1
        )

        # Computing a separate softmax for each goal of each other agent
        chunked_other_beliefs = others_beliefs.split(self.goal_size, dim=-1)

        # shape: [batch, agents, agents-1, sum(goal_size)]
        others_beliefs = torch.cat(
            [
                torch.nn.functional.softmax(chunk, dim=-1)
                for chunk in chunked_other_beliefs
            ],
            dim=-1,
        )

        actor_logits, actor_memory = super().forward(
            observations,
            actor_rnn_hidden,
            latents=beliefs.unsqueeze(-2)
            .repeat_interleave(agents - 1, -2)
            .unsqueeze(-2),
        )

        return actor_logits, actor_memory, beliefs, others_beliefs

    def update(self, buffer: MPE_ReplayBuffer, labels):

        # labels is a tensor of shape [batch, seq, goal_size]
        # it contains the ground-truth environment labels
        # for each goal over which agents produce beliefs

        # shape: [batch, seq, agents, obs_dim]
        observations = (
            torch.tensor(buffer.buffer["states"])[..., : self.obs_dim]
            .float()
            .to(self.device)
        )
        batch, seq, agents, _ = observations.shape

        actor_h = torch.zeros(batch * agents, 64, device=self.device)

        loss = 0.0
        for t in range(seq):
            x = self.fc(observations[:, t].flatten(0, 1))
            actor_h = self.rnn(self.mem_norm(x), actor_h)
            beliefs = self.belief_net(actor_h.view(batch, agents, -1))
            start = 0
            for goal in self.goal_size:
                pred_chunk = beliefs[..., start : start + goal]
                target_chunk = (
                    labels[:, t, start : start + goal]
                    .unsqueeze(-2)
                    .repeat_interleave(agents, -2)
                )
                loss += torch.nn.functional.cross_entropy(pred_chunk, target_chunk)

        self.belief_net_optimizer.zero_grad()
        loss.backward()
        self.belief_net_optimizer.step()

        return {"belief_loss": loss}


class SupTomActor(MPE_Actor):

    def __init__(
        self,
        config: Config,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(config=config)

        self.device = device
        self.obs_dim = config.env.obs_dim
        self.num_agents = config.env.num_agents
        self.goal_size = config.module.goal_size

        self.belief_net = MLP(
            input_size=self.obs_dim, output_size=sum(config.module.goal_size)
        )

        self.others_belief_net = MLP(
            input_size=config.training.actor.no_memory_fc_layers[-1],
            output_size=(config.env.num_agents - 1) * sum(config.module.goal_size),
        )

        self.belief_net_optimizer = torch.optim.Adam(
            self.belief_net.parameters(), lr=config.module.belief_lr
        )

    def forward(self, observations):

        if len(observations.size()) == 3:
            batch, agents, features = observations.shape
        elif len(observations.size()) == 4:
            batch, seq, agents, features = observations.shape

        # Agent own beliefs, shape: [batch, agents, sum(goal_size)]
        beliefs = self.belief_net(observations)

        # Computing a separate softmax for each goal
        chunked_beliefs = beliefs.split(self.goal_size, dim=-1)
        beliefs = torch.cat(
            [torch.nn.functional.softmax(chunk, dim=-1) for chunk in chunked_beliefs],
            dim=-1,
        )

        actor_features = self.actor(observations)
        others_beliefs = self.others_belief_net(actor_features)
        if len(observations.size()) == 3:
            others_beliefs = others_beliefs.view(
                batch, self.num_agents, self.num_agents - 1, -1
            )
        elif len(observations.size()) == 4:
            others_beliefs = others_beliefs.view(
                batch, seq, self.num_agents, self.num_agents - 1, -1
            )

        # Computing a separate softmax for each goal of each other agent
        chunked_other_beliefs = others_beliefs.split(self.goal_size, dim=-1)

        # shape: [batch, agents, agents-1, sum(goal_size)]
        others_beliefs = torch.cat(
            [
                torch.nn.functional.softmax(chunk, dim=-1)
                for chunk in chunked_other_beliefs
            ],
            dim=-1,
        )

        actor_logits = super().forward(
            observations,
            latents=beliefs.unsqueeze(-2)
            .repeat_interleave(self.num_agents - 1, -2)
            .unsqueeze(-2),
        )

        return actor_logits, beliefs, others_beliefs

    def update(self, buffer: MPE_ReplayBuffer, labels):

        # labels is a tensor of shape [batch, seq, goal_size]
        # it contains the ground-truth environment labels
        # for each goal over which agents produce beliefs

        # shape: [batch, seq, agents, obs_dim]
        observations = (
            torch.tensor(buffer.buffer["states"])[..., : self.obs_dim]
            .float()
            .to(self.device)
        )
        batch, seq, agents, _ = observations.shape

        beliefs = self.belief_net(observations)

        loss = 0.0
        start = 0
        for goal in self.goal_size:
            pred_chunk = beliefs[..., start : start + goal]
            target_chunk = (
                labels[..., start : start + goal]
                .unsqueeze(-2)
                .repeat_interleave(agents, -2)
            )
            loss += torch.nn.functional.cross_entropy(pred_chunk, target_chunk)

        self.belief_net_optimizer.zero_grad()
        loss.backward()
        self.belief_net_optimizer.step()

        return {"belief_loss": loss}
