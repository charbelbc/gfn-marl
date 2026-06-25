import torch
from common.nets import MLP
from common.utils import MPE_ReplayBuffer
from common.config import Config


class EMAMultiCodebookQuantizer(torch.nn.Module):

    def __init__(self, state_size, dict_size, emb_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.state_size = state_size
        self.dict_size = dict_size
        self.emb_dim = emb_dim
        self.decay = decay
        self.eps = eps

        self.embedding = torch.nn.Parameter(
            torch.randn(self.state_size, self.dict_size, self.emb_dim)
        )
        self.register_buffer(
            "ema_cluster_size", torch.zeros(self.state_size, self.dict_size)
        )
        self.register_buffer(
            "ema_w", torch.randn(self.state_size, self.dict_size, self.emb_dim)
        )

    def forward(self, z, training: bool = True):
        batch, state_size, emb_dim = z.shape
        assert state_size == self.state_size and emb_dim == self.emb_dim
        z_exp = z.unsqueeze(2)
        emb_exp = self.embedding.unsqueeze(0)
        dist = torch.sum((z_exp - emb_exp) ** 2, dim=-1)
        indices = torch.argmin(dist, dim=-1)
        z_q = torch.gather(
            self.embedding.unsqueeze(0).expand(batch, -1, -1, -1),
            2,
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, emb_dim),
        ).squeeze(2)
        if training:
            one_hot = torch.nn.functional.one_hot(indices, self.dict_size).float()
            cluster_size = one_hot.sum(dim=0)
            dw = torch.einsum("bsk,bsd->skd", one_hot, z)
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
            )
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
            n = self.ema_cluster_size.sum(dim=-1, keepdim=True)
            cluster_size = (
                (self.ema_cluster_size + self.eps) / (n + self.dict_size * self.eps) * n
            )
            self.embedding.data = self.ema_w / cluster_size.unsqueeze(-1)
        loss = torch.nn.functional.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()

        return z_q, loss, indices


class SequentialVQVAE(torch.nn.Module):

    def __init__(
        self,
        config: Config,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        self.device = device
        self.action_dim = config.action_dim
        self.obs_dim = config.obs_dim
        self.dict_size = config.vqvae_dict_size
        self.state_size = config.vqvae_state_size

        self.action_encoder = MLP(
            input_size=config.action_dim, output_size=16, hidden_sizes=[32]
        ).to(self.device)
        self.observation_encoder = MLP(
            input_size=config.obs_dim + config.num_agents,
            output_size=32,
            hidden_sizes=[32, 32],
            with_feature_norm=True,
            with_layer_norm=True,
        ).to(self.device)
        self.reward_encoder = MLP(
            input_size=1,
            output_size=16,
            hidden_sizes=[16],
            with_feature_norm=True,
            with_layer_norm=True,
        ).to(self.device)
        self.encoder_lstm = torch.nn.GRUCell(32 + 16 + 16 * config.num_agents, 64).to(
            self.device
        )
        self.encoder = MLP(64, self.state_size, [64])
        self.quantizer = EMAMultiCodebookQuantizer(self.state_size, self.dict_size, 1)

        self.latent_layer = MLP(
            input_size=config.vqvae_state_size, output_size=32, hidden_sizes=[32]
        ).to(self.device)
        self.decoder_lstm = torch.nn.GRU(32, 32, batch_first=True).to(self.device)
        self.action_decoder = MLP(
            input_size=32, output_size=config.action_dim, hidden_sizes=[64]
        ).to(self.device)

        params = (
            list(self.action_encoder.parameters())
            + list(self.observation_encoder.parameters())
            + list(self.reward_encoder.parameters())
            + list(self.encoder_lstm.parameters())
            + list(self.encoder.parameters())
            + list(self.quantizer.parameters())
            + list(self.latent_layer.parameters())
            + list(self.decoder_lstm.parameters())
            + list(self.action_decoder.parameters())
        )

        self.vqvae_optimizer = torch.optim.Adam(params, lr=config.vqvae_lr)

    def update(self, buffer: MPE_ReplayBuffer):

        # shape: [batch, seq, agents, obs_dim]
        observations = torch.tensor(buffer.buffer["states"])[..., : self.obs_dim]
        batch, seq, agents, _ = observations.shape

        # shape: [batch, seq, agents, agents-1, obs_dim]
        observations = (
            torch.cat(
                [observations, torch.zeros(batch, 1, agents, observations.shape[-1])],
                dim=1,
            )
            .float()
            .unsqueeze(-2)
            .repeat_interleave(agents - 1, -2)
            .to(self.device)
        )
        ids = (
            torch.eye(agents, device=self.device)
            .unsqueeze(0)
            .repeat(agents, 1, 1)[
                ~torch.eye(agents, dtype=torch.bool, device=self.device)
            ]
            .view(agents, agents - 1, agents)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, seq + 1, -1, -1, -1)
        )
        observations = torch.cat([observations, ids], dim=-1)

        # shape: [batch, seq, agents, action_dim]
        actions = torch.tensor(buffer.buffer["actions"])
        actions = (
            torch.cat([torch.zeros(batch, 1, agents), actions], dim=1)
            .long()
            .to(self.device)
        )
        actions = torch.nn.functional.one_hot(actions, self.action_dim).float()

        # shape: [batch, seq, agents, 1]
        rewards = torch.tensor(buffer.buffer["rewards"])
        rewards = (
            torch.cat([torch.zeros(batch, 1, agents), rewards], dim=1)
            .unsqueeze(-1)
            .float()
            .to(self.device)
        )

        # shape of each [batch, seq, agents, agents-1, features]
        obs_features = self.observation_encoder(observations)
        reward_features = (
            self.reward_encoder(rewards).unsqueeze(-2).repeat_interleave(agents - 1, -2)
        )
        action_features = self.action_encoder(actions)
        action_features = (
            action_features.unsqueeze(2).repeat_interleave(agents, 2).flatten(3)
        )
        action_features = action_features.unsqueeze(-2).repeat_interleave(
            agents - 1, -2
        )

        # shape [batch, seq, agents, agents-1, features]
        encoder_inputs = torch.cat(
            [obs_features, action_features, reward_features], dim=-1
        )

        enc_h = torch.zeros(batch * agents * (agents - 1), 64, device=self.device)
        dec_loss, quantization_loss = 0.0, 0.0

        for t in range(seq):
            enc_h = self.encoder_lstm(encoder_inputs[:, t].flatten(0, -2), enc_h)
            z = (
                self.encoder(enc_h)
                .view(batch * agents * (agents - 1), self.state_size, 1)
                .to(self.device)
            )
            z_q, q_loss, indices = self.quantizer(z)
            quantization_loss += q_loss
            z_q = z_q.view(batch, agents, agents - 1, self.state_size)
            indices = indices.view(batch, agents, agents - 1, self.state_size)

            # shape: [batch, agents, agents-1, features]
            dec_h = self.latent_layer(z_q)

            rec = self.decoder_lstm(
                obs_features[:, t:-1].permute(0, 2, 3, 1, 4).flatten(0, 2),
                dec_h.flatten(0, 2).unsqueeze(0),
            )[0].view(batch, agents, agents - 1, seq - t, -1)

            # shape: [batch, agents, agents-1, seq-t, action_dim]
            logprobs = self.action_decoder(rec)

            # shape: [batch, seq-t, agents, agents-1]
            targets = (
                actions[:, t + 1 :]
                .argmax(dim=-1)
                .unsqueeze(2)
                .expand(batch, seq - t, agents, agents)[
                    :,
                    :,
                    ~torch.eye(agents, dtype=torch.bool, device=actions.device),
                ]
                .view(batch, seq - t, agents, agents - 1)
            ).permute(0, 2, 3, 1)

            dec_loss += torch.nn.functional.cross_entropy(
                logprobs.flatten(0, -2),
                targets.flatten().long(),
                reduction="sum",
            )

        loss = (dec_loss + quantization_loss) / batch
        self.vqvae_optimizer.zero_grad()
        loss.backward()
        self.vqvae_optimizer.step()

        return {
            "dec_loss": (dec_loss / batch).item(),
            "q_loss": (quantization_loss / batch).item(),
        }

    def sample_latents(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        encoder_h: torch.Tensor,
    ):

        batch, agents, _ = observations.shape

        assert encoder_h.shape == (batch, agents, agents - 1, 64)

        # shape: [batch, agents, agents-1, obs_dim]
        observations = (
            observations.float()
            .unsqueeze(-2)
            .repeat_interleave(agents - 1, -2)
            .to(self.device)
        )
        ids = (
            torch.eye(agents, device=self.device)
            .unsqueeze(0)
            .repeat(agents, 1, 1)[
                ~torch.eye(agents, dtype=torch.bool, device=self.device)
            ]
            .view(agents, agents - 1, agents)
            .unsqueeze(0)
            .expand(batch, -1, -1, -1)
        )
        observations = torch.cat([observations, ids], dim=-1)

        # shape: [batch, agents, action_dim]
        actions = actions.long().to(self.device)
        actions = torch.nn.functional.one_hot(actions, self.action_dim).float()

        # shape: [batch, agents, 1]
        rewards = rewards.unsqueeze(-1).float().to(self.device)

        # shape of each [batch, seq, agents, agents-1, features]
        obs_features = self.observation_encoder(observations)
        reward_features = (
            self.reward_encoder(rewards).unsqueeze(-2).repeat_interleave(agents - 1, -2)
        )
        action_features = self.action_encoder(actions)
        action_features = (
            action_features.unsqueeze(1).repeat_interleave(agents, 1).flatten(2)
        )
        action_features = action_features.unsqueeze(-2).repeat_interleave(
            agents - 1, -2
        )

        # shape [batch, agents, agents-1, features]
        encoder_inputs = torch.cat(
            [obs_features, action_features, reward_features], dim=-1
        )

        next_encoder_h = self.encoder_lstm(
            encoder_inputs.flatten(0, -2),
            encoder_h.flatten(0, 2).to(self.device),
        )

        z = (
            self.encoder(next_encoder_h)
            .view(batch * agents * (agents - 1), self.state_size, 1)
            .to(self.device)
        )
        z_q, _, indices = self.quantizer(z)
        z_q = z_q.view(batch, agents, agents - 1, 1, self.state_size)
        indices = indices.view(batch, agents, agents - 1, 1, self.state_size)

        return (
            z_q.detach(),
            indices.detach(),
            next_encoder_h.view(batch, agents, agents - 1, -1),
        )
