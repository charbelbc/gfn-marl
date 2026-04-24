import torch
import numpy as np
from typing import Tuple

import gymnasium as gym
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.envs.babyai.core.verifier import (
    GoToInstr,
    ObjDesc,
    PickupInstr,
    PutNextInstr,
)
from minigrid.wrappers import FullyObsWrapper
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
import copy

from minigrid.core.world_object import Ball, Box, Door, Key, WorldObj
from minigrid.core.grid import Grid, Wall


class Envvv(RoomGridLevel):

    def __init__(self, **kwargs):
        super().__init__(num_rows=1, num_cols=1, **kwargs)

    def gen_mission(self):
        self.grid.set(6, 5, Wall())
        self.grid.set(1, 2, Wall())
        self.grid.set(6, 2, Wall())
        self.grid.set(1, 5, Wall())
        self.place_obj(Box("green"), (6, 6), (1, 1))
        self.place_obj(Box("blue"), (1, 1), (1, 1))
        self.place_obj(Box("purple"), (1, 6), (1, 1))
        self.place_obj(Box("yellow"), (6, 1), (1, 1))
        self.place_agent(0, 0)
        room = self.room_from_pos(0, 0)
        t = True
        while t:
            try:
                for obj in room.objs:
                    self.grid.set(obj.cur_pos[0], obj.cur_pos[1], None)
                room.objs = []
                if np.random.rand() < 0.25:
                    self.add_object(0, 0, "ball", "blue")
                if np.random.rand() < 0.25:
                    self.add_object(0, 0, "ball", "green")
                if np.random.rand() < 0.25:
                    self.add_object(0, 0, "ball", "purple")
                if np.random.rand() < 0.25:
                    self.add_object(0, 0, "ball", "yellow")
                if len(room.objs) == 0:
                    self.add_object(
                        0,
                        0,
                        "ball",
                        self._rand_elem(["blue", "green", "purple", "yellow"]),
                    )
                self.check_objs_reachable()
                t = False
            except RejectSampling:
                continue
        self.target = np.random.choice(
            [obj.color for obj in room.objs] + ["switch"],
            p=np.concatenate([2 * np.ones((len(room.objs))), np.array([1])])
            / (2 * len(room.objs) + 1),
        )
        if self.target != "switch":
            self.current_color = self.target
        elif self.target == "switch":
            self.current_color = self._rand_elem([obj.color for obj in room.objs])
        self.instrs = PutNextInstr(
            ObjDesc("ball", self.current_color), ObjDesc("box", self.current_color)
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        room = self.room_from_pos(0, 0)
        if not terminated:
            if np.random.rand() < 0.5 and len(room.objs) < 4:
                color = self._rand_elem(
                    list(
                        np.array(["blue", "green", "purple", "yellow"])[
                            ~np.isin(
                                ["blue", "green", "purple", "yellow"],
                                [obj.color for obj in room.objs],
                            )
                        ]
                    )
                )
                t = True
                while t:
                    try:
                        for obj in room.objs:
                            if obj.color == color:
                                self.grid.set(obj.cur_pos[0], obj.cur_pos[1], None)
                                room.objs.remove(obj)
                        self.add_object(0, 0, "ball", color)
                        self.check_objs_reachable()
                        t = False
                    except RejectSampling:
                        continue
        if terminated:
            for obj in room.objs:
                if obj.color == self.current_color:
                    self.grid.set(obj.cur_pos[0], obj.cur_pos[1], None)
                    room.objs.remove(obj)
            if self.target == "switch":
                if self.current_color == "green":
                    self.current_color = "blue"
                elif self.current_color == "blue":
                    self.current_color = "green"
                elif self.current_color == "purple":
                    self.current_color = "yellow"
                elif self.current_color == "yellow":
                    self.current_color = "purple"
            self.instrs = PutNextInstr(
                ObjDesc("ball", self.current_color), ObjDesc("box", self.current_color)
            )
            self.instrs.reset_verifier(self)
        return obs, reward, terminated, truncated, info


class EEnvvv(RoomGridLevel):

    def __init__(self, **kwargs):
        super().__init__(num_rows=1, num_cols=1, **kwargs)

    def gen_mission(self):
        self.grid.set(6, 5, Wall())
        self.grid.set(1, 2, Wall())
        self.grid.set(6, 2, Wall())
        self.grid.set(1, 5, Wall())
        self.place_obj(Box("green"), (6, 6), (1, 1))
        self.place_obj(Box("blue"), (1, 1), (1, 1))
        self.place_obj(Box("purple"), (1, 6), (1, 1))
        self.place_obj(Box("yellow"), (6, 1), (1, 1))
        self.place_agent(0, 0)
        room = self.room_from_pos(0, 0)
        t = True
        while t:
            try:
                for obj in room.objs:
                    self.grid.set(obj.cur_pos[0], obj.cur_pos[1], None)
                room.objs = []
                self.add_object(0, 0, "ball", "blue")
                self.add_object(0, 0, "ball", "green")
                self.add_object(0, 0, "ball", "purple")
                self.add_object(0, 0, "ball", "yellow")
                self.check_objs_reachable()
                t = False
            except RejectSampling:
                continue
        self.target = np.random.choice(
            ["blue", "green", "purple", "yellow", "switch"],
            p=[1 / 6, 1 / 6, 1 / 6, 1 / 6, 2 / 6],
        )
        if self.target != "switch":
            self.current_color = self.target
        elif self.target == "switch":
            self.current_color = self._rand_elem(["blue", "green", "purple", "yellow"])
        self.instrs = PutNextInstr(
            ObjDesc("ball", self.current_color), ObjDesc("box", self.current_color)
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            room = self.room_from_pos(0, 0)
            t = True
            while t:
                try:
                    for obj in [obj for obj in room.objs]:
                        self.grid.set(obj.cur_pos[0], obj.cur_pos[1], None)
                    room.objs = []
                    self.add_object(0, 0, "ball", "blue")
                    self.add_object(0, 0, "ball", "green")
                    self.add_object(0, 0, "ball", "purple")
                    self.add_object(0, 0, "ball", "yellow")
                    self.check_objs_reachable()
                    if self.target == "switch":
                        if self.current_color == "green":
                            self.current_color = "blue"
                        elif self.current_color == "blue":
                            self.current_color = "green"
                        elif self.current_color == "purple":
                            self.current_color = "yellow"
                        elif self.current_color == "yellow":
                            self.current_color = "purple"
                    self.instrs = PutNextInstr(
                        ObjDesc("ball", self.current_color),
                        ObjDesc("box", self.current_color),
                    )
                    self.instrs.reset_verifier(self)
                    t = False
                except RejectSampling:
                    continue
        return obs, reward, terminated, truncated, info


class EMGFlowNet:

    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dict_size: int = 4,
        state_size: int = 5,
    ):

        self.device = device
        self.n_actions = state_size * dict_size + 1

        self.dict_size = dict_size
        self.state_size = state_size

        self.emb = ImageBOWEmbedding(147, 128).to(self.device)
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(7, 7), stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        ).to(self.device)
        self.encoder_lstm = torch.nn.GRUCell(128, 64).to(self.device)
        cond_features = 64

        self.pf = MLP(
            input_size=(dict_size + 1) * state_size,
            output_size=cond_features,
            hidden_sizes=[64, 64],
        ).to(self.device)

        self.pf_final = MLP(
            input_size=2 * cond_features,
            output_size=self.n_actions,
            hidden_sizes=[256, 256],
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

        # self.logz = MLP(input_size=cond_features, output_size=1, hidden_sizes=[512]).to(
        #     self.device
        # )
        self.logz = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.gfn_optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.emb.parameters())
                    + list(self.convnet.parameters())
                    + list(self.encoder_lstm.parameters())
                    + list(self.pf.parameters())
                    + list(self.pf_final.parameters()),
                    # + list(self.pb.parameters())
                    # + list(self.pb_final.parameters()),
                    "lr": 0.001,
                },
                {
                    "params": self.logz,
                    "lr": 0.01,
                },
            ]
        )

        self.dec_emb = ImageBOWEmbedding(147, 128).to(self.device)
        self.dec_convnet = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(7, 7), stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        ).to(self.device)
        self.codebook = torch.nn.Parameter(
            torch.randn(state_size, dict_size, 1, device=self.device),
            requires_grad=True,
        ).to(self.device)
        self.latent_mlp = MLP(state_size, 2 * 128, hidden_sizes=[64, 64]).to(
            self.device
        )
        self.decoder = MLP(128, 6, hidden_sizes=[128, 128]).to(self.device)

        self.decoder_optimizer = torch.optim.Adam(
            params=list(self.dec_emb.parameters())
            + list(self.dec_convnet.parameters())
            + [self.codebook]
            + list(self.latent_mlp.parameters())
            + list(self.decoder.parameters()),
            lr=0.0003,
        )

    def preprocess_states(self, states: torch.Tensor) -> torch.Tensor:

        one_hot = torch.nn.functional.one_hot(states, num_classes=self.dict_size + 1)

        return one_hot.reshape(
            *states.shape[:-1], self.state_size * (self.dict_size + 1)
        )

    def update_masks(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

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

    def train_gfn(self, observations, actions, rand_prob):

        observations = observations[:, :-1].to(self.device)

        batch, seq, *channels = observations.shape

        obs_features = self.convnet(
            self.emb(observations.reshape(batch * seq, *channels))
        ).reshape(batch, seq, -1)

        dec_features = self.dec_convnet(
            self.dec_emb(observations.reshape(batch * seq, *channels))
        ).reshape(batch, seq, -1)

        enc_h = torch.zeros(batch, 64, device=observations.device)

        gfn_loss = torch.zeros(batch, seq - 1)

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(obs_features[:, t], enc_h)

            # forward_terminal_states, forward_log_pf, forward_log_pb = (
            #     self.sample_trajectories(conditioning=enc_h, rand_prob=rand_prob)
            # )

            forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                conditioning=enc_h, rand_prob=rand_prob
            )

            with torch.no_grad():
                z = self.codebook[
                    torch.arange(self.state_size, device=self.device)
                    .unsqueeze(0)
                    .expand(batch, self.state_size),
                    forward_terminal_states,
                ].view(batch, -1)
                dec_latents = self.latent_mlp(z.to(self.device))
                gamma, beta = dec_latents.chunk(2, dim=-1)
                gamma = 1 + gamma
                film_out = dec_features[:, t:] * gamma.unsqueeze(1).repeat(
                    1, seq - t, 1
                ) + beta.unsqueeze(1).repeat(1, seq - t, 1)
                logprobs = self.decoder(torch.nn.functional.relu(film_out))[
                    :, : seq - t - 1
                ]

            rewards = (
                -torch.nn.functional.cross_entropy(
                    logprobs.flatten(0, 1),
                    actions[:, t:].flatten(0, 1),
                    reduction="none",
                )
                .reshape(batch, seq - t - 1)
                .sum(-1)
                .to(self.device)
                / torch.linspace(10, 0.2, seq - 1)[t]
            )

            # log_z = self.logz(enc_h).squeeze()

            gfn_loss[:, t] = (self.logz + forward_log_pf - rewards).pow(2)

        self.gfn_optimizer.zero_grad()
        gfn_loss.mean().backward()
        self.gfn_optimizer.step()

        return gfn_loss.mean().item()

    def train_decoder(self, observations, actions):

        observations = observations[:, :-1].to(self.device)

        batch, seq, *channels = observations.shape

        obs_features = self.convnet(
            self.emb(observations.reshape(batch * seq, *channels))
        ).reshape(batch, seq, -1)

        dec_features = self.dec_convnet(
            self.dec_emb(observations.reshape(batch * seq, *channels))
        ).reshape(batch, seq, -1)

        enc_h = torch.zeros(batch, 64, device=observations.device)

        dec_loss = 0.0

        for t in range(seq - 1):

            with torch.no_grad():

                enc_h = self.encoder_lstm(obs_features[:, t], enc_h)

                forward_terminal_states, forward_log_pf, forward_log_pb = (
                    self.sample_trajectories(conditioning=enc_h)
                )

            # dec_latents = self.latent_mlp(
            #     torch.nn.functional.one_hot(forward_terminal_states, self.dict_size)
            #     .reshape(batch, self.state_size * self.dict_size)
            #     .float()
            #     .to(self.device)
            # )
            z = self.codebook[
                torch.arange(self.state_size, device=self.device)
                .unsqueeze(0)
                .expand(batch, self.state_size),
                forward_terminal_states,
            ].view(batch, -1)
            dec_latents = self.latent_mlp(z.to(self.device))
            gamma, beta = dec_latents.chunk(2, dim=-1)
            gamma = 1 + gamma
            film_out = (
                dec_features[:, t:]
                + dec_features[:, t:] * gamma.unsqueeze(1).repeat(1, seq - t, 1)
                + beta.unsqueeze(1).repeat(1, seq - t, 1)
            )
            logprobs = self.decoder(torch.nn.functional.relu(film_out))[
                :, : seq - t - 1
            ]

            # logprobs = self.decoder(
            #     torch.cat(
            #         [
            #             dec_features[:, t:],
            #             dec_latents.unsqueeze(1).repeat(1, seq - t, 1),
            #         ],
            #         dim=-1,
            #     )
            # )[:, : seq - t - 1]

            dec_loss += torch.nn.functional.cross_entropy(
                logprobs.flatten(0, 1),
                actions[:, t:].flatten(0, 1),
                reduction="sum",
            )

        self.decoder_optimizer.zero_grad()
        dec_loss.backward()
        self.decoder_optimizer.step()

        return dec_loss.mean().item()


class SequentialVQVAE(torch.nn.Module):

    def __init__(self, state_size, dict_size, embedding_size):
        super().__init__()

        self.state_size = state_size
        self.dict_size = dict_size
        self.embedding_size = embedding_size

        self.decay = 0.99
        self.eps = 1e-5

        self.emb = ImageBOWEmbedding(147, 128)
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(7, 7), stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.encoder_lstm = torch.nn.GRUCell(128, 64)
        self.latent_pre = torch.nn.Linear(64, state_size * embedding_size)
        codebook = torch.randn(state_size, dict_size, embedding_size)
        codebook.uniform_(-1 / dict_size, 1 / dict_size)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(state_size, dict_size))
        self.register_buffer("embedding_avg", codebook.clone())
        self.latent_mlp = MLP(state_size * embedding_size, 2 * 128, hidden_sizes=[128])
        self.decoder = MLP(128, 6, hidden_sizes=[128, 128])

    def forward(self, observations_l, actions):

        observations = observations_l[:, :-1]
        batch, seq, *channels = observations.shape

        obs_features = self.convnet(
            self.emb(observations.reshape(batch * seq, *channels))
        ).reshape(batch, seq, -1)

        dec_loss, e_latent_loss, q_latent_loss = 0.0, 0.0, 0.0
        enc_h = torch.zeros(batch, 64, device=observations.device)

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(obs_features[:, t], enc_h)
            z = self.latent_pre(enc_h).view(batch, self.state_size, self.embedding_size)
            indices = (
                (z.unsqueeze(2) - self.codebook.unsqueeze(0)).pow(2).sum(-1).argmin(-1)
            )
            # z_q = self.codebook(indices)
            z_q = self.codebook[
                torch.arange(self.state_size, device=z.device)
                .unsqueeze(0)
                .expand(batch, self.state_size),
                indices,
            ]
            one_hot = torch.nn.functional.one_hot(
                indices.flatten(), self.dict_size
            ).float()
            cluster_size = one_hot.sum(0)
            embed_sum = one_hot.t() @ z.view(-1, self.embedding_size)
            self.cluster_size.data.mul_(self.decay).add_(
                cluster_size, alpha=1 - self.decay
            )
            self.embedding_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )
            n = self.cluster_size.sum(dim=1, keepdim=True)
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.dict_size * self.eps) * n
            )
            self.codebook.data.copy_(self.embedding_avg / cluster_size.unsqueeze(-1))

            e_latent_loss += torch.nn.functional.mse_loss(
                z_q.detach(), z, reduction="mean"
            )
            # q_latent_loss += torch.nn.functional.mse_loss(
            #     z_q, z.detach(), reduction="mean"
            # )

            z_q = z + (z_q - z).detach()  # straight through estimation
            z_q = z_q.view(batch, -1)

            gamma, beta = self.latent_mlp(z_q).chunk(2, dim=-1)
            gamma = 1 + gamma
            film_out = obs_features[:, t:] * gamma.unsqueeze(1).repeat(
                1, seq - t, 1
            ) + beta.unsqueeze(1).repeat(1, seq - t, 1)

            out = self.decoder(torch.nn.functional.relu(film_out))[:, : seq - t - 1]

            dec_loss += torch.nn.functional.cross_entropy(
                out.flatten(0, 1), actions[:, t:].flatten(0, 1), reduction="sum"
            )

        return dec_loss / batch, e_latent_loss / batch


class SequentialVAE(torch.nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.emb = ImageBOWEmbedding(147, 128)
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(7, 7), stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.encoder_lstm = torch.nn.GRUCell(128, 64)
        self.latent_mu = torch.nn.Linear(64, latent_dim)
        self.latent_logvar = torch.nn.Linear(64, latent_dim)

        self.latent_mlp = MLP(latent_dim, 2 * 128, hidden_sizes=[64, 64])
        self.decoder = MLP(128, 6, hidden_sizes=[128, 128])

    def forward(self, observations_l, actions):

        observations = observations_l[:, :-1]
        lengths = observations_l[:, -1, 0, 0, 0]

        batch, seq, *channels = observations.shape

        obs_features = self.convnet(
            self.emb(observations.reshape(batch * seq, *channels))
        ).reshape(batch, seq, -1)

        dec_loss, kl_loss = 0.0, 0.0
        enc_h = torch.zeros(batch, 64, device=observations.device)

        mu_prior = torch.zeros(batch, self.latent_dim, device=observations.device)
        logvar_prior = torch.zeros_like(mu_prior)

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(obs_features[:, t], enc_h)
            mu_q = self.latent_mu(enc_h)
            logvar_q = self.latent_logvar(enc_h)
            z_t = mu_q + torch.randn_like(logvar_q) * torch.exp(0.5 * logvar_q)

            dec_latents = self.latent_mlp(z_t)

            gamma, beta = dec_latents.chunk(2, dim=-1)
            gamma = 1 + gamma
            film_out = obs_features[:, t:] * gamma.unsqueeze(1).repeat(
                1, seq - t, 1
            ) + beta.unsqueeze(1).repeat(1, seq - t, 1)

            out = self.decoder(torch.nn.functional.relu(film_out))[:, : seq - t - 1]

            dec_loss += torch.nn.functional.cross_entropy(
                out.flatten(0, 1), actions[:, t:].flatten(0, 1), reduction="sum"
            )

            kl_loss += 0.5 * torch.sum(
                logvar_prior
                - logvar_q
                + (torch.exp(logvar_q) + (mu_q - mu_prior) ** 2)
                / torch.exp(logvar_prior)
                - 1,
                dim=-1,
            ).sum(-1)

            mu_prior, logvar_prior = mu_q, logvar_q

        return dec_loss / batch, kl_loss / batch


class ImageBOWEmbedding(torch.nn.Module):
    def __init__(self, max_value, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(3 * max_value, embedding_dim)

    def forward(self, inputs):
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(
            inputs.device
        )
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


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


def generate_conditioning_sampless(n_samples: int = 10, with_frames: bool = False):

    observations, actions, lengths, missions = [], [], [], []
    all_actions = []
    env = Envvv(render_mode="rgb_array", max_steps=100)
    env = FullyObsWrapper(env)

    if with_frames:
        frames = []

    for _ in range(n_samples):
        observation, act, other_act = [], [], []
        c_mission = []
        if with_frames:
            frame = []
        obs, info = env.reset()
        env1 = copy.deepcopy(env)
        expert = BabyAIBot(env)
        last_action = None
        terminated, truncated = False, False
        observation.append(torch.from_numpy(obs["image"]).transpose(0, 2))
        # missions.append(env.mission.split()[-2:])
        # missions.append(env.target)
        if with_frames:
            frame.append(env.render())
        c_mission.append(env.target)
        while not truncated:
            c_mission.append(env.current_color)
            if len(env.unwrapped.instrs.desc_move.obj_set) == 0:
                action = 5
                env.instrs.reset_verifier(env)
                expert = BabyAIBot(env)
                last_action = None
            else:
                action = expert.replan(last_action)
                last_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            act.append(action.real)
            observation.append(torch.from_numpy(obs["image"]).transpose(0, 2))
            if with_frames:
                frame.append(env.render())
            if terminated:
                expert = BabyAIBot(env)
                last_action = None
        lengths.append(len(observation))
        observations.append(torch.stack(observation))
        actions.append(torch.tensor(act))
        missions.append(c_mission)
        if with_frames:
            frames.append(frame)

    observations = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
    observations = torch.cat(
        [
            observations[..., :8, :8],
            torch.tensor(lengths)
            .view(n_samples, 1, 1, 1, 1)
            .expand(n_samples, 1, 3, observations.shape[-2], observations.shape[-1]),
        ],
        dim=1,
    )

    if with_frames:
        return observations, actions, all_actions, lengths, np.array(missions), frames
    else:
        return observations, actions, all_actions, lengths, np.array(missions)
