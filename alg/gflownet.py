import torch
from common.nets import MLP
from common.utils import MPE_ReplayBuffer
from common.config import Config


class EMGFlowNet:

    def __init__(
        self,
        config: Config,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):

        self.device = device
        self.n_actions = config.gfn_state_size * config.gfn_dict_size + 1
        self.rand_prob = config.gfn_rand_prob
        self.greedy_decoder = config.gfn_greedy_decoder
        self.action_dim = config.action_dim
        self.single_codebook = config.gfn_single_codebook
        self.obs_dim = config.obs_dim

        self.dict_size = config.gfn_dict_size
        self.state_size = config.gfn_state_size

        self.encoder_steps = config.gfn_encoder_steps
        self.decoder_steps = config.gfn_decoder_steps

        self.action_encoder = MLP(
            input_size=config.action_dim, output_size=16, hidden_sizes=[32]
        ).to(self.device)
        self.observation_encoder = MLP(
            input_size=config.obs_dim + config.num_agents,
            output_size=32,
            hidden_sizes=[32, 32],
        ).to(self.device)
        self.reward_encoder = MLP(input_size=1, output_size=16, hidden_sizes=[16]).to(
            self.device
        )
        self.encoder_lstm = torch.nn.GRUCell(32 + 16 + 16 * config.num_agents, 64).to(
            self.device
        )
        enc_params = (
            list(self.action_encoder.parameters())
            + list(self.observation_encoder.parameters())
            + list(self.reward_encoder.parameters())
            + list(self.encoder_lstm.parameters())
        )

        self.pf = MLP(
            input_size=(config.gfn_dict_size + 1) * config.gfn_state_size,
            output_size=64,
            hidden_sizes=[64, 64],
        ).to(self.device)

        self.pf_final = MLP(
            input_size=2 * 64,
            output_size=self.n_actions,
            hidden_sizes=[64, 64],
        ).to(self.device)

        enc_params += list(self.pf.parameters()) + list(self.pf_final.parameters())

        if config.gfn_use_pb:
            self.pb = MLP(
                input_size=(config.gfn_dict_size + 1) * config.gfn_state_size,
                output_size=64,
                hidden_sizes=[64, 64],
            ).to(self.device)

            self.pb_final = MLP(
                input_size=2 * 64,
                output_size=self.n_actions - 1,
                hidden_sizes=[64, 64],
            ).to(self.device)

            enc_params += list(self.pb.parameters()) + list(self.pb_final.parameters())

        self.logz = MLP(input_size=64, output_size=1, hidden_sizes=[64]).to(self.device)

        self.gfn_optimizer = torch.optim.Adam(
            [
                {
                    "params": enc_params,
                    "lr": config.gfn_lr,
                },
                {
                    "params": self.logz.parameters(),
                    "lr": config.gfn_logz_lr,
                },
            ]
        )

        if config.gfn_single_codebook:
            self.codebook = torch.nn.Embedding(config.gfn_dict_size, 1).to(self.device)
        else:
            self.codebook = torch.nn.Parameter(
                torch.randn(
                    config.gfn_state_size, config.gfn_dict_size, 1, device=self.device
                ),
                requires_grad=True,
            ).to(self.device)
        self.decoder_obs_encoder = MLP(
            input_size=config.obs_dim + config.num_agents,
            output_size=32,
            hidden_sizes=[32, 32],
        ).to(self.device)
        self.latent_layer = MLP(
            input_size=config.gfn_state_size, output_size=32, hidden_sizes=[32]
        ).to(self.device)
        self.decoder_lstm = torch.nn.GRU(32, 32, batch_first=True)
        self.action_decoder = MLP(
            input_size=32, output_size=config.action_dim, hidden_sizes=[64]
        )

        self.decoder_optimizer = torch.optim.Adam(
            params=(
                (
                    list(self.codebook.parameters())
                    if config.gfn_single_codebook
                    else [self.codebook]
                )
                + list(self.decoder_obs_encoder.parameters())
                + list(self.latent_layer.parameters())
                + list(self.decoder_lstm.parameters())
                + list(self.action_decoder.parameters())
            ),
            lr=config.gfn_dec_lr,
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

    def train_gflownet(self, buffer: MPE_ReplayBuffer):

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

        # shape [batch, seq, agents, agents-1, features]
        dec_features = (self.decoder_obs_encoder(observations))[:, :-1]

        enc_h = torch.zeros(batch * agents * (agents - 1), 64, device=self.device)
        gfn_loss = torch.zeros(batch * agents * (agents - 1), seq)

        for t in range(seq):

            enc_h = self.encoder_lstm(encoder_inputs[:, t].flatten(0, -2), enc_h)

            forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                conditioning=enc_h, rand_prob=self.rand_prob
            )

            with torch.no_grad():
                if self.single_codebook:
                    z = self.codebook(
                        forward_terminal_states.view(
                            batch, agents, agents - 1, self.state_size
                        ).to(self.device)
                    ).squeeze(-1)
                else:
                    z = self.codebook[
                        torch.arange(self.state_size, device=self.device)
                        .unsqueeze(0)
                        .expand(batch * agents * (agents - 1), self.state_size),
                        forward_terminal_states.to(self.device),
                    ].view(batch, agents, agents - 1, self.state_size)

                # shape: [batch, agents, agents-1, features]
                dec_h = self.latent_layer(z)

                rec = self.decoder_lstm(
                    dec_features[:, t:].permute(0, 2, 3, 1, 4).flatten(0, 2),
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

                grewards = (
                    -torch.nn.functional.cross_entropy(
                        logprobs.flatten(0, -2),
                        targets.flatten().long(),
                        reduction="none",
                    )
                    .reshape(batch, agents, agents - 1, seq - t)
                    .sum(-1)
                    .to(self.device)
                    / torch.linspace(seq, 1, seq)[t]
                )

                log_z = self.logz(enc_h).squeeze()

            gfn_loss[:, t] = (log_z + forward_log_pf - grewards.flatten()).pow(2)

        self.gfn_optimizer.zero_grad()
        gfn_loss.mean().backward()
        self.gfn_optimizer.step()

        return gfn_loss.mean().item()

    def train_decoder(self, buffer: MPE_ReplayBuffer):

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

        # shape [batch, seq, agents, agents-1, features]
        dec_features = (self.decoder_obs_encoder(observations))[:, :-1]

        enc_h = torch.zeros(batch * agents * (agents - 1), 64, device=self.device)

        dec_loss = 0.0

        for t in range(seq):

            with torch.no_grad():

                enc_h = self.encoder_lstm(encoder_inputs[:, t].flatten(0, -2), enc_h)

                forward_terminal_states, _ = self.sample_ar_trajectories(
                    conditioning=enc_h,
                    rand_prob=0,
                    prob_exponent=-1 if self.greedy_decoder else 1,
                )

            if self.single_codebook:
                z = self.codebook(
                    forward_terminal_states.view(
                        batch, agents, agents - 1, self.state_size
                    ).to(self.device)
                ).squeeze(-1)
            else:
                z = self.codebook[
                    torch.arange(self.state_size, device=self.device)
                    .unsqueeze(0)
                    .expand(batch * agents * (agents - 1), self.state_size),
                    forward_terminal_states.to(self.device),
                ].view(batch, agents, agents - 1, self.state_size)

            # shape: [batch, agents, agents-1, features]
            dec_h = self.latent_layer(z)

            rec = self.decoder_lstm(
                dec_features[:, t:].permute(0, 2, 3, 1, 4).flatten(0, 2),
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

            dec_loss += (
                torch.nn.functional.cross_entropy(
                    logprobs.flatten(0, -2),
                    targets.flatten().long(),
                    reduction="sum",
                )
                / batch
            )

        self.decoder_optimizer.zero_grad()
        dec_loss.backward()
        self.decoder_optimizer.step()

        return dec_loss.item()

    def sample_latents(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        gflownet_encoder_h: torch.Tensor,
        rand_prob: float = 0,
        prob_exponent: float = 1,
    ) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch, agents, _ = observations.shape

        assert gflownet_encoder_h.shape == (batch, agents, agents - 1, 64)

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

        next_gflownet_h = self.encoder_lstm(
            encoder_inputs.flatten(0, -2),
            gflownet_encoder_h.flatten(0, 2).to(self.device),
        )

        forward_terminal_states, _ = self.sample_ar_trajectories(
            conditioning=next_gflownet_h,
            rand_prob=rand_prob,
            prob_exponent=prob_exponent,
        )

        if self.single_codebook:
            z = self.codebook(
                forward_terminal_states.view(
                    batch, agents, agents - 1, self.state_size
                ).to(self.device)
            ).squeeze(-1)
        else:
            z = self.codebook[
                torch.arange(self.state_size, device=self.device)
                .unsqueeze(0)
                .expand(batch * agents * (agents - 1), self.state_size),
                forward_terminal_states.to(self.device),
            ].view(batch, agents, agents - 1, self.state_size)

        return (
            z.detach(),
            forward_terminal_states.view(
                batch, agents, agents - 1, self.state_size
            ).detach(),
            next_gflownet_h.view(batch, agents, agents - 1, -1),
        )

    def update(self, buffer: MPE_ReplayBuffer):

        for _ in range(self.encoder_steps):
            gfn_loss = self.train_gflownet(buffer)

        for _ in range(self.decoder_steps):
            decoder_loss = self.train_decoder(buffer)

        return {"gfn_loss": gfn_loss, "gfn_decoder_loss": decoder_loss}
