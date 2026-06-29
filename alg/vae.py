import torch
from common.nets import MLP
from common.utils import MPE_ReplayBuffer
from common.config import Config


class SequentialVAE(torch.nn.Module):

    def __init__(
        self,
        config: Config,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        self.device = device
        self.action_dim = config.env.action_dim
        self.obs_dim = config.env.obs_dim
        self.latent_dim = config.module.vae_latent_size
        self.kl_factor = config.module.vae_kl_factor

        self.action_encoder = MLP(
            input_size=config.env.action_dim, output_size=16, hidden_sizes=[32]
        ).to(self.device)
        self.observation_encoder = MLP(
            input_size=config.env.obs_dim + config.env.num_agents,
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
        self.encoder_lstm = torch.nn.GRUCell(
            32 + 16 + 16 * config.env.num_agents, 64
        ).to(self.device)
        self.latent_mu = MLP(64, self.latent_dim, [64]).to(self.device)
        self.latent_logvar = MLP(64, self.latent_dim, [64]).to(self.device)

        self.latent_layer = MLP(
            input_size=self.latent_dim, output_size=32, hidden_sizes=[32]
        ).to(self.device)
        self.decoder_lstm = torch.nn.GRU(32, 32, batch_first=True).to(self.device)
        self.action_decoder = MLP(
            input_size=32, output_size=config.env.action_dim, hidden_sizes=[64]
        ).to(self.device)

        params = (
            list(self.action_encoder.parameters())
            + list(self.observation_encoder.parameters())
            + list(self.reward_encoder.parameters())
            + list(self.encoder_lstm.parameters())
            + list(self.latent_mu.parameters())
            + list(self.latent_logvar.parameters())
            + list(self.latent_layer.parameters())
            + list(self.decoder_lstm.parameters())
            + list(self.action_decoder.parameters())
        )

        self.vae_optimizer = torch.optim.Adam(params, lr=config.module.vae_lr)

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

        mu_prior = torch.zeros(
            batch * agents * (agents - 1), self.latent_dim, device=self.device
        )
        logvar_prior = torch.zeros_like(mu_prior)

        dec_loss, kl_loss = 0.0, 0.0

        for t in range(seq):

            enc_h = self.encoder_lstm(encoder_inputs[:, t].flatten(0, -2), enc_h)

            mu_m = self.latent_mu(enc_h)
            logvar_m = self.latent_logvar(enc_h)
            z = mu_m + torch.randn_like(logvar_m) * torch.exp(0.5 * logvar_m)

            z = z.view(batch, agents, agents - 1, self.latent_dim)

            # shape: [batch, agents, agents-1, features]
            dec_h = self.latent_layer(z)

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

            kl_loss += 0.5 * torch.sum(
                logvar_prior
                - logvar_m
                + (torch.exp(logvar_m) + (mu_m - mu_prior) ** 2)
                / torch.exp(logvar_prior)
                - 1,
                dim=-1,
            ).sum(-1)

            mu_prior, logvar_prior = mu_m, logvar_m

        loss = (dec_loss + self.kl_factor * kl_loss) / batch
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

        return {
            "dec_loss": (dec_loss / batch).item(),
            "q_loss": (kl_loss / batch).item(),
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

        mu_m = self.latent_mu(next_encoder_h)
        logvar_m = self.latent_logvar(next_encoder_h)
        z = mu_m + torch.randn_like(logvar_m) * torch.exp(0.5 * logvar_m)

        z = z.view(batch, agents, agents - 1, 1, self.latent_dim)

        return (
            z.detach(),
            next_encoder_h.view(batch, agents, agents - 1, -1),
        )
