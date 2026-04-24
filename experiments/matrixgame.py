import torch
import numpy as np


class MatrixGame:

    def __init__(self):
        self.rewards = {(0, 0): 1, (1, 0): 8, (0, 1): 0, (1, 1): -1}

    def reset(self, agent_type: int | None = None):
        if agent_type is not None:
            self.agent_type = agent_type
        else:
            self.agent_type = np.random.randint(1, 4)
        self.consecutive_swerve = 0

    def step(self, action: int):
        # Actions: 0 -> swerve, 1 -> straight
        agent_action = 1
        if self.consecutive_swerve == self.agent_type:
            self.consecutive_swerve = 0
            agent_action = 0
        if action == 0 and agent_action == 1:
            self.consecutive_swerve += 1
        reward = self.rewards[(action, agent_action)]
        return agent_action, reward


class EMGFlowNet:

    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dict_size: int = 4,
        state_size: int = 5,
    ):

        state_size = 2 * state_size
        self.device = device
        self.n_actions = state_size * dict_size + 1

        self.dict_size = dict_size
        self.state_size = state_size

        self.action_encoder = MLP(2, 16, [16])
        self.reward_encoder = MLP(1, 16, [16])
        self.encoder_lstm = torch.nn.GRUCell(32, 64)

        self.pf = MLP(
            input_size=(dict_size + 1) * state_size,
            output_size=64,
            hidden_sizes=[64, 64],
        ).to(self.device)

        self.pf_final = MLP(
            input_size=2 * 64,
            output_size=self.n_actions,
            hidden_sizes=[64, 64],
        ).to(self.device)

        # self.pb = MLP(
        #     input_size=(dict_size + 1) * state_size,
        #     output_size=cond_features,
        #     hidden_sizes=[64, 64],
        # ).to(self.device)

        # self.pb_final = MLP(
        #     input_size=2 * cond_features,
        #     output_size=self.n_actions - 1,
        #     hidden_sizes=[256, 256],
        # ).to(self.device)

        self.logz = MLP(input_size=64, output_size=1, hidden_sizes=[64]).to(self.device)
        # self.logz = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.gfn_optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.action_encoder.parameters())
                    + list(self.reward_encoder.parameters())
                    + list(self.encoder_lstm.parameters())
                    + list(self.pf.parameters())
                    + list(self.pf_final.parameters()),
                    # + list(self.pb.parameters())
                    # + list(self.pb_final.parameters()),
                    "lr": 0.0002,
                },
                {
                    "params": self.logz.parameters(),
                    "lr": 0.01,
                },
            ]
        )

        self.latent_layer = MLP(int(state_size / 2), 32, [32])
        self.latent_layer_t = MLP(int(state_size / 2) + 32, 64, [64])
        self.decoder_lstm = torch.nn.GRU(32, 64, batch_first=True)
        self.action_decoder = MLP(64, 1, [64])
        # self.codebook = torch.nn.Parameter(
        #     torch.randn(state_size, dict_size, 1, device=self.device),
        #     requires_grad=True,
        # ).to(self.device)
        self.codebook = torch.nn.Embedding(dict_size, 1).to(self.device)

        self.decoder_optimizer = torch.optim.Adam(
            params=list(self.latent_layer.parameters())
            + list(self.latent_layer_t.parameters())
            # + [self.codebook]
            + list(self.codebook.parameters())
            + list(self.decoder_lstm.parameters())
            + list(self.action_decoder.parameters()),
            lr=0.001,
        )

    def _get_name(self):
        return "EMGFlowNet"

    def preprocess_states(self, states: torch.Tensor) -> torch.Tensor:

        one_hot = torch.nn.functional.one_hot(states, num_classes=self.dict_size + 1)

        return one_hot.reshape(
            *states.shape[:-1], self.state_size * (self.dict_size + 1)
        )

    def update_masks(self, states: torch.Tensor) -> list[torch.Tensor, torch.Tensor]:

        forward_masks = torch.ones((states.shape[0], self.n_actions), dtype=bool)
        backward_masks = torch.ones((states.shape[0], self.n_actions - 1), dtype=bool)

        forward_masks[..., :-1] = (states == 0).repeat_interleave(self.dict_size, -1)
        forward_masks[..., -1] = (states != 0).sum(-1) == self.state_size

        backward_masks = (states != 0).repeat_interleave(self.dict_size, -1)

        return forward_masks, backward_masks

    def forward_step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:

        return states.scatter(
            -1,
            actions.div(self.dict_size, rounding_mode="floor"),
            actions.fmod(self.dict_size) + 1,
        )

    def backward_step(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:

        return states.scatter(-1, actions.div(self.dict_size, rounding_mode="floor"), 0)

    def sample_backward_trajectories(
        self, conditioning: torch.Tensor, final_states: torch.Tensor
    ):

        n_trajectories = final_states.shape[0]
        states = (final_states + 1).clone().cpu()
        dones = torch.zeros(n_trajectories, dtype=bool)

        log_pf = torch.zeros(n_trajectories, device=self.device)
        log_pb = torch.zeros(n_trajectories, device=self.device)

        actions = torch.tensor([self.n_actions - 1], device=self.device).repeat(
            (n_trajectories, 1)
        )

        while not dones.all():

            c_states = states[~dones]
            c_cond = conditioning[~dones]

            forward_logits = self.pf_final(
                torch.cat(
                    [
                        self.pf(
                            self.preprocess_states(c_states).float().to(self.device)
                        ),
                        c_cond.to(self.device),
                    ],
                    dim=-1,
                )
            )
            backward_logits = self.pb_final(
                torch.cat(
                    [
                        self.pb(
                            self.preprocess_states(c_states).float().to(self.device)
                        ),
                        c_cond.to(self.device),
                    ],
                    dim=-1,
                )
            )
            forward_mask, backward_mask = self.update_masks(c_states)
            forward_logits = forward_logits.masked_fill(
                ~forward_mask.to(self.device), -float("inf")
            ).log_softmax(-1)
            backward_logits = backward_logits.masked_fill(
                ~backward_mask.to(self.device), -float("inf")
            ).log_softmax(-1)

            log_pf[~dones] += torch.gather(
                forward_logits, dim=-1, index=actions.reshape(c_states.shape[0], 1)
            ).squeeze()

            dones[~dones] = (c_states.sum(-1) == 0).cpu()
            backward_probs = backward_logits[c_states.sum(-1) != 0].softmax(-1)
            actions = backward_probs.multinomial(1)
            log_pb[~dones] += torch.gather(
                backward_logits[c_states.sum(-1) != 0], dim=-1, index=actions
            ).squeeze()

            states[~dones] = self.backward_step(
                states[~dones], actions.cpu().reshape(states[~dones].shape[0], 1)
            )

        return log_pf, log_pb

    def sample_ar_trajectories(
        self,
        conditioning: torch.Tensor,
        rand_prob: float = 0.0,
        prob_exponent: float = 1.0,
    ):

        n_trajectories = conditioning.shape[0]
        states = torch.zeros(n_trajectories, self.state_size).long()
        mask = torch.zeros(
            n_trajectories,
            self.state_size * self.dict_size,
            dtype=bool,
            device=self.device,
        )
        log_pf = torch.zeros(n_trajectories, device=self.device)

        for step in range(self.state_size):
            forward_logits = self.pf_final(
                torch.cat(
                    [
                        self.pf(self.preprocess_states(states).float().to(self.device)),
                        conditioning.to(self.device),
                    ],
                    dim=-1,
                )
            )[:, :-1]
            mask[:, step * self.dict_size : (step + 1) * self.dict_size] = True
            forward_probs = forward_logits.masked_fill(
                ~mask.to(self.device), -float("inf")
            ).softmax(-1)
            if prob_exponent > 0:
                gfn_actions = (forward_probs**prob_exponent).multinomial(1)
            else:
                gfn_actions = forward_probs.argmax(-1, keepdim=True)
            rand_update = (
                torch.rand((forward_probs.shape[0], 1), device=forward_probs.device)
                < rand_prob
            ).long()
            actions = (1 - rand_update) * gfn_actions + rand_update * (
                torch.ones_like(forward_probs, device=self.device)
                * mask.to(self.device)
            ).multinomial(1)
            log_pf += torch.gather(forward_logits, dim=-1, index=actions).squeeze()
            states = self.forward_step(
                states, actions.cpu().reshape(states.shape[0], 1)
            )
            mask[:, step * self.dict_size : (step + 1) * self.dict_size] = False

        return states - 1, log_pf

    def sample_trajectories(
        self,
        conditioning: torch.Tensor,
        rand_prob: float = 0.0,
        prob_exponent: float = 1.0,
    ):

        n_trajectories = conditioning.shape[0]

        states = torch.zeros(n_trajectories, self.state_size).long()
        dones = torch.zeros(n_trajectories, dtype=bool)

        log_pf = torch.zeros(n_trajectories, device=self.device)
        log_pb = torch.zeros(n_trajectories, device=self.device)
        actions = None

        while not dones.all():

            c_states = states[~dones]
            c_cond = conditioning[~dones]

            forward_logits = self.pf_final(
                torch.cat(
                    [
                        self.pf(
                            self.preprocess_states(c_states).float().to(self.device)
                        ),
                        c_cond.to(self.device),
                    ],
                    dim=-1,
                )
            )
            backward_logits = self.pb_final(
                torch.cat(
                    [
                        self.pb(
                            self.preprocess_states(c_states).float().to(self.device)
                        ),
                        c_cond.to(self.device),
                    ],
                    dim=-1,
                )
            )
            forward_mask, backward_mask = self.update_masks(c_states)
            forward_logits = forward_logits.masked_fill(
                ~forward_mask.to(self.device), -float("inf")
            ).log_softmax(-1)
            backward_logits = backward_logits.masked_fill(
                ~backward_mask.to(self.device), -float("inf")
            ).log_softmax(-1)

            if actions is not None:
                log_pb[~dones] += torch.gather(
                    backward_logits, dim=-1, index=actions.reshape(c_states.shape[0], 1)
                ).squeeze()

            forward_probs = forward_logits.softmax(-1)
            if prob_exponent > 0:
                gfn_actions = (forward_probs**prob_exponent).multinomial(1)
            else:
                gfn_actions = forward_probs.argmax(-1, keepdim=True)
            rand_update = (
                torch.rand((forward_probs.shape[0], 1), device=forward_probs.device)
                < rand_prob
            ).long()
            actions = (1 - rand_update) * gfn_actions + rand_update * (
                torch.ones_like(forward_probs, device=self.device)
                * forward_mask.to(self.device)
            ).multinomial(1)
            log_pf[~dones] += torch.gather(
                forward_logits, dim=-1, index=actions
            ).squeeze()

            dones[~dones] = (actions.squeeze() == self.n_actions - 1).cpu()
            actions = actions[actions.squeeze() != self.n_actions - 1]
            states[~dones] = self.forward_step(
                states[~dones], actions.cpu().reshape(states[~dones].shape[0], 1)
            )

        return states - 1, log_pf, log_pb

    def train_gfn(self, actions, rewards, rand_prob):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        reward_features = self.reward_encoder(rewards)

        enc_h = torch.zeros(batch, 64, device=actions.device)

        gfn_loss = torch.zeros(batch, seq)

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(
                torch.cat([action_features[:, t], reward_features[:, t]], dim=-1), enc_h
            )

            forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                conditioning=enc_h, rand_prob=rand_prob
            )

            with torch.no_grad():
                # z = self.codebook[
                #     torch.arange(self.state_size, device=self.device)
                #     .unsqueeze(0)
                #     .expand(batch, self.state_size),
                #     forward_terminal_states,
                # ].view(batch, -1)
                z = self.codebook(forward_terminal_states).view(batch, -1)
                m, m_t = z.chunk(2, dim=-1)
                dec_features = self.latent_layer(m.view(batch, -1))
                dec_h = self.latent_layer_t(
                    torch.cat([dec_features, m_t.view(batch, -1)], dim=-1)
                )
                rec = self.decoder_lstm(
                    dec_features.unsqueeze(1).repeat_interleave(seq - t - 1, dim=1),
                    dec_h.unsqueeze(0),
                )[0]
                logprobs = self.action_decoder(torch.nn.functional.relu(rec))

            grewards = (
                -torch.nn.functional.binary_cross_entropy_with_logits(
                    logprobs.flatten(),
                    actions[:, t + 1 :, 1].flatten(0, 1),
                    reduction="none",
                )
                .reshape(batch, seq - t - 1)
                .sum(-1)
                .to(self.device)
                / torch.linspace(seq - 1, 1, seq - 1)[t]
            )

            log_z = self.logz(enc_h).squeeze()

            gfn_loss[:, t] = (log_z + forward_log_pf - grewards).pow(2)

        self.gfn_optimizer.zero_grad()
        gfn_loss.mean().backward()
        self.gfn_optimizer.step()

        return gfn_loss.mean().item()

    def train_decoder(self, actions, rewards, prob_exponent):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        reward_features = self.reward_encoder(rewards)

        enc_h = torch.zeros(batch, 64, device=actions.device)

        dec_loss = 0.0

        for t in range(seq - 1):

            with torch.no_grad():

                enc_h = self.encoder_lstm(
                    torch.cat([action_features[:, t], reward_features[:, t]], dim=-1),
                    enc_h,
                )

                forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                    conditioning=enc_h, prob_exponent=prob_exponent
                )

            # z = self.codebook[
            #     torch.arange(self.state_size, device=self.device)
            #     .unsqueeze(0)
            #     .expand(batch, self.state_size),
            #     forward_terminal_states,
            # ].view(batch, -1)
            z = self.codebook(forward_terminal_states).view(batch, -1)
            m, m_t = z.chunk(2, dim=-1)
            dec_features = self.latent_layer(m.view(batch, -1))
            dec_h = self.latent_layer_t(
                torch.cat([dec_features, m_t.view(batch, -1)], dim=-1)
            )
            rec = self.decoder_lstm(
                dec_features.unsqueeze(1).repeat_interleave(seq - t, dim=1),
                dec_h.unsqueeze(0),
            )[0]
            logprobs = self.action_decoder(torch.nn.functional.relu(rec))

            dec_loss += (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logprobs.flatten(),
                    actions[:, t:, 1].flatten(0, 1),
                    reduction="sum",
                )
                / batch
            )

        self.decoder_optimizer.zero_grad()
        dec_loss.backward()
        self.decoder_optimizer.step()

        return dec_loss.mean().item()

    def sample_latents(self, actions, rewards, rand_prob, prob_exponent):
        batch, seq, _ = actions.shape
        action_features = self.action_encoder(actions)
        reward_features = self.reward_encoder(rewards)
        enc_h = torch.zeros(batch, 64, device=actions.device)
        perm_latents, var_latents = [], []
        perm_ind, var_ind = [], []

        with torch.no_grad():
            for t in range(seq - 1):
                enc_h = self.encoder_lstm(
                    torch.cat([action_features[:, t], reward_features[:, t]], dim=-1),
                    enc_h,
                )
                forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                    conditioning=enc_h, rand_prob=rand_prob, prob_exponent=prob_exponent
                )
                # z = self.codebook[
                #     torch.arange(self.state_size, device=self.device)
                #     .unsqueeze(0)
                #     .expand(batch, self.state_size),
                #     forward_terminal_states,
                # ].view(batch, -1)
                z = self.codebook(forward_terminal_states).view(batch, -1)
                m, m_t = z.chunk(2, dim=-1)
                m_indices, m_t_indices = forward_terminal_states.chunk(2, dim=-1)
                perm_latents.append(m.view(batch, -1))
                var_latents.append(m_t.view(batch, -1))
                perm_ind.append(m_indices)
                var_ind.append(m_t_indices)
        perm_latents = torch.stack(perm_latents, dim=1)
        var_latents = torch.stack(var_latents, dim=1)
        perm_ind = torch.stack(perm_ind, dim=1)
        var_ind = torch.stack(var_ind, dim=1)
        return perm_latents, var_latents, perm_ind, var_ind


class SequentialVQVAE(torch.nn.Module):
    def __init__(
        self,
        state_size=2,
        dict_size=8,
        emb_dim=1,
    ):
        super().__init__()
        self.state_size, self.dict_size, self.emb_dim = state_size, dict_size, emb_dim
        self.action_encoder = MLP(2, 16, [16])
        self.reward_encoder = MLP(1, 16, [16])
        self.encoder_lstm = torch.nn.GRUCell(32, 64)
        self.bottom_encoder = MLP(64, state_size * emb_dim, [64])
        self.top_encoder = MLP(state_size * emb_dim, state_size * emb_dim, [64])
        self.top_quantizer = EMAMultiCodebookQuantizer(state_size, dict_size, emb_dim)
        self.bottom_quantizer = EMAMultiCodebookQuantizer(
            state_size, dict_size, emb_dim
        )
        self.latent_layer = MLP(state_size * emb_dim, 32, [32])
        self.latent_layer_t = MLP(state_size * emb_dim + 32, 64, [64])
        self.decoder_lstm = torch.nn.GRU(32, 64, batch_first=True)
        self.action_decoder = MLP(64, 1, [64])

    def forward(self, actions, rewards):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        reward_features = self.reward_encoder(rewards)

        dec_loss, top_quantization_loss, bottom_quantization_loss = 0.0, 0.0, 0.0
        enc_h = torch.zeros(batch, 64, device=actions.device)
        perm_latents, var_latents = [], []
        perm_ind, var_ind = [], []

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(
                torch.cat([action_features[:, t], reward_features[:, t]], dim=-1), enc_h
            )
            bottom_z = self.bottom_encoder(enc_h).view(
                batch, self.state_size, self.emb_dim
            )
            top_z = self.top_encoder(bottom_z.view(batch, -1)).view(
                batch, self.state_size, self.emb_dim
            )
            m_t, loss_t, m_t_ind = self.top_quantizer(top_z)
            m, loss_b, m_ind = self.bottom_quantizer(bottom_z)
            top_quantization_loss += loss_t
            bottom_quantization_loss += loss_b

            dec_features = self.latent_layer(m.view(batch, -1))
            dec_h = self.latent_layer_t(
                torch.cat([dec_features, m_t.view(batch, -1)], dim=-1)
            )
            rec = self.decoder_lstm(
                dec_features.unsqueeze(1).repeat_interleave(seq - t - 1, dim=1),
                dec_h.unsqueeze(0),
            )[0]
            out = self.action_decoder(torch.nn.functional.relu(rec))
            dec_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                out.flatten(), actions[:, t + 1 :, 1].flatten(0, 1), reduction="sum"
            )
            perm_latents.append(m.view(batch, -1))
            var_latents.append(m_t.view(batch, -1))
            perm_ind.append(m_ind)
            var_ind.append(m_t_ind)
        perm_latents = torch.stack(perm_latents, dim=1)
        var_latents = torch.stack(var_latents, dim=1)
        perm_ind = torch.stack(perm_ind, dim=1)
        var_ind = torch.stack(var_ind, dim=1)

        return (
            dec_loss / batch,
            (top_quantization_loss + bottom_quantization_loss) / batch,
            perm_latents,
            var_latents,
            perm_ind,
            var_ind,
        )


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


class SequentialVAE(torch.nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_encoder = MLP(2, 16, [16])
        self.reward_encoder = MLP(1, 16, [16])
        self.encoder_lstm = torch.nn.GRUCell(32, 64)
        self.latent_mu = MLP(64, latent_dim, [64])
        self.latent_logvar = MLP(64, latent_dim, [64])
        self.encoder_layer = MLP(64, 64, [64])
        self.latent_mu_t = MLP(64, latent_dim, [64])
        self.latent_logvar_t = MLP(64, latent_dim, [64])

        self.latent_layer = MLP(latent_dim, 32, [32])
        self.latent_layer_t = MLP(latent_dim + 32, 64, [64])
        self.decoder_lstm = torch.nn.GRU(32, 64, batch_first=True)
        self.action_decoder = MLP(64, 1, [64])

    def forward(self, actions, rewards):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        reward_features = self.reward_encoder(rewards)

        dec_loss, kl_loss = 0.0, 0.0
        enc_h = torch.zeros(batch, 64, device=actions.device)

        mu_prior = torch.zeros(batch, 2 * self.latent_dim, device=actions.device)
        logvar_prior = torch.zeros_like(mu_prior)
        perm_latents, var_latents = [], []

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(
                torch.cat([action_features[:, t], reward_features[:, t]], dim=-1), enc_h
            )
            mu_m = self.latent_mu(enc_h)
            logvar_m = self.latent_logvar(enc_h)
            m = mu_m + torch.randn_like(logvar_m) * torch.exp(0.5 * logvar_m)
            feat = self.encoder_layer(enc_h)
            mu_mt = self.latent_mu_t(feat)
            logvar_mt = self.latent_logvar_t(feat)
            m_t = mu_mt + torch.randn_like(logvar_mt) * torch.exp(0.5 * logvar_mt)

            dec_features = self.latent_layer(m)
            dec_h = self.latent_layer_t(torch.cat([dec_features, m_t], dim=-1))

            rec = self.decoder_lstm(
                dec_features.unsqueeze(1).repeat_interleave(seq - t - 1, dim=1),
                dec_h.unsqueeze(0),
            )[0]
            out = self.action_decoder(torch.nn.functional.relu(rec))

            dec_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                out.flatten(), actions[:, t + 1 :, 1].flatten(0, 1), reduction="sum"
            )

            kl_loss += 0.5 * torch.sum(
                logvar_prior
                - torch.cat([logvar_m, logvar_mt], dim=-1)
                + (
                    torch.exp(torch.cat([logvar_m, logvar_mt], dim=-1))
                    + (torch.cat([mu_m, mu_mt], dim=-1) - mu_prior) ** 2
                )
                / torch.exp(logvar_prior)
                - 1,
                dim=-1,
            ).sum(-1)

            mu_prior, logvar_prior = torch.cat(
                [logvar_m, logvar_mt], dim=-1
            ), torch.cat([mu_m, mu_mt], dim=-1)

            perm_latents.append(m)
            var_latents.append(m_t)
        perm_latents = torch.stack(perm_latents, dim=1)
        var_latents = torch.stack(var_latents, dim=1)

        return dec_loss / batch, kl_loss / batch, perm_latents, var_latents


class MLP(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list = [512, 512],
        activation=torch.nn.ReLU,
    ):

        super().__init__()

        _net = [torch.nn.Linear(input_size, hidden_sizes[0]), activation()]

        for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            _net += [torch.nn.Linear(h1, h2), activation()]

        if len(hidden_sizes) > 1:
            _net += [torch.nn.Linear(h2, output_size)]
        else:
            _net += [torch.nn.Linear(hidden_sizes[-1], output_size)]

        self.net = torch.nn.Sequential(*_net)

    def forward(self, x):
        return self.net(x)
