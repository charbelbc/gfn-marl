import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

STAY, UP, DOWN, LEFT, RIGHT, PICK, DROP = range(7)
BLUE = 1
GREEN = 2
YELLOW = 3
PURPLE = 4
SWITCH1 = 5
SWITCH2 = 6


class SLDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = latents
        self.labels = labels

    def __getitem__(self, index):
        return (self.latents[index], self.labels[index], index)

    def __len__(self):
        return len(self.latents)


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


class GridWorld:
    def __init__(self, size=5, seed=None):
        self.size = size
        self.rng = np.random.RandomState(seed)

        # Bank positions
        self.green_bank = np.array([0, 0])
        self.blue_bank = np.array([size - 1, size - 1])
        self.yellow_bank = np.array([size - 1, 0])
        self.purple_bank = np.array([0, size - 1])

        self.reset()

    def reset(self):
        self.agent_pos = np.array([self.size // 2, self.size // 2])
        self.carrying = None

        self.blue_coin = None
        self.green_coin = None
        self.yellow_coin = None
        self.purple_coin = None

        self.spawn_coin(BLUE)
        self.spawn_coin(GREEN)
        self.spawn_coin(YELLOW)
        self.spawn_coin(PURPLE)

        return self._get_obs()

    def spawn_coin(self, coin_type):
        if coin_type == BLUE and self.blue_coin is None:
            self.blue_coin = self._random_empty_cell()
        elif coin_type == GREEN and self.green_coin is None:
            self.green_coin = self._random_empty_cell()
        elif coin_type == YELLOW and self.yellow_coin is None:
            self.yellow_coin = self._random_empty_cell()
        elif coin_type == PURPLE and self.purple_coin is None:
            self.purple_coin = self._random_empty_cell()

    def _random_empty_cell(self):
        while True:
            pos = np.array(
                [self.rng.randint(0, self.size), self.rng.randint(0, self.size)]
            )

            if np.array_equal(pos, self.agent_pos):
                continue
            if self.blue_coin is not None and np.array_equal(pos, self.blue_coin):
                continue
            if self.green_coin is not None and np.array_equal(pos, self.green_coin):
                continue
            if self.yellow_coin is not None and np.array_equal(pos, self.yellow_coin):
                continue
            if self.purple_coin is not None and np.array_equal(pos, self.purple_coin):
                continue
            if np.array_equal(pos, self.green_bank):
                continue
            if np.array_equal(pos, self.blue_bank):
                continue
            if np.array_equal(pos, self.yellow_bank):
                continue
            if np.array_equal(pos, self.purple_bank):
                continue

            return pos

    def step(self, action):
        reward = 0

        # --- Movement ---
        if action == UP:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == DOWN:
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == LEFT:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == RIGHT:
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        # --- Pick ---
        elif action == PICK:
            if self.carrying is None:
                if self.blue_coin is not None and np.array_equal(
                    self.agent_pos, self.blue_coin
                ):
                    self.carrying = BLUE
                    self.blue_coin = None
                elif self.green_coin is not None and np.array_equal(
                    self.agent_pos, self.green_coin
                ):
                    self.carrying = GREEN
                    self.green_coin = None
                elif self.yellow_coin is not None and np.array_equal(
                    self.agent_pos, self.yellow_coin
                ):
                    self.carrying = YELLOW
                    self.yellow_coin = None
                elif self.purple_coin is not None and np.array_equal(
                    self.agent_pos, self.purple_coin
                ):
                    self.carrying = PURPLE
                    self.purple_coin = None

        # --- Drop ---
        elif action == DROP:
            if self.carrying == BLUE and np.array_equal(self.agent_pos, self.blue_bank):
                reward = 1
                self.carrying = None
                self.spawn_coin(BLUE)
            elif self.carrying == GREEN and np.array_equal(
                self.agent_pos, self.green_bank
            ):
                reward = 1
                self.carrying = None
                self.spawn_coin(GREEN)
            elif self.carrying == YELLOW and np.array_equal(
                self.agent_pos, self.yellow_bank
            ):
                reward = 1
                self.carrying = None
                self.spawn_coin(YELLOW)
            elif self.carrying == PURPLE and np.array_equal(
                self.agent_pos, self.purple_bank
            ):
                reward = 1
                self.carrying = None
                self.spawn_coin(PURPLE)

        return self._get_obs(), reward

    def _get_obs(self):
        return {
            "agent_pos": self.agent_pos.copy(),
            "carrying": self.carrying,
            "blue_coin": None if self.blue_coin is None else self.blue_coin.copy(),
            "green_coin": None if self.green_coin is None else self.green_coin.copy(),
            "yellow_coin": (
                None if self.yellow_coin is None else self.yellow_coin.copy()
            ),
            "purple_coin": (
                None if self.purple_coin is None else self.purple_coin.copy()
            ),
        }

    def get_array_obs(self):
        return np.concatenate(
            [
                self.agent_pos.copy() / self.size,
                [self.carrying / 4 if self.carrying is not None else -1],
                (
                    self.blue_coin / self.size
                    if self.blue_coin is not None
                    else list((-1, -1))
                ),
                (
                    self.green_coin / self.size
                    if self.green_coin is not None
                    else list((-1, -1))
                ),
                (
                    self.yellow_coin / self.size
                    if self.yellow_coin is not None
                    else list((-1, -1))
                ),
                (
                    self.purple_coin / self.size
                    if self.purple_coin is not None
                    else list((-1, -1))
                ),
            ]
        )

    def render(self):
        grid = np.full((self.size, self.size), ".", dtype=object)

        # Banks
        gr, gc = self.green_bank
        br, bc = self.blue_bank
        yr, yc = self.yellow_bank
        pr, pc = self.purple_bank
        grid[gr, gc] = "G"
        grid[br, bc] = "B"
        grid[yr, yc] = "Y"
        grid[pr, pc] = "P"

        # Coins
        if self.blue_coin is not None:
            r, c = self.blue_coin
            grid[r, c] = "b"
        if self.green_coin is not None:
            r, c = self.green_coin
            grid[r, c] = "g"
        if self.yellow_coin is not None:
            r, c = self.yellow_coin
            grid[r, c] = "y"
        if self.purple_coin is not None:
            r, c = self.purple_coin
            grid[r, c] = "p"

        # Agent
        r, c = self.agent_pos
        grid[r, c] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print(f"Carrying: {self.carrying}")
        print()


class Agent:
    def __init__(self, type):
        self.type = type
        self.switch1 = type == SWITCH1
        self.switch2 = type == SWITCH2
        if self.type == SWITCH1:
            self.type = np.random.randint(1, 3)
        if self.type == SWITCH2:
            self.type = np.random.randint(3, 5)

    def act(self, env):
        obs = env._get_obs()
        target_coin = (
            "blue_coin"
            if self.type == BLUE
            else (
                "green_coin"
                if self.type == GREEN
                else "yellow_coin" if self.type == YELLOW else "purple_coin"
            )
        )
        if obs["carrying"] == self.type:
            target_bank = (
                env.blue_bank
                if self.type == BLUE
                else (
                    env.green_bank
                    if self.type == GREEN
                    else env.yellow_bank if self.type == YELLOW else env.purple_bank
                )
            )
            if np.array_equal(obs["agent_pos"], target_bank):
                if self.switch1:
                    self.type = BLUE if self.type == GREEN else GREEN
                if self.switch2:
                    self.type = YELLOW if self.type == PURPLE else PURPLE
                return DROP
            return self._move_to_target(obs["agent_pos"], target_bank)
        if obs[target_coin] is not None:
            if np.array_equal(obs["agent_pos"], obs[target_coin]):
                return PICK
            return self._move_to_target(obs["agent_pos"], obs[target_coin])
        return STAY

    def _move_to_target(self, pos, target):
        r, c = pos
        tr, tc = target
        if r < tr:
            return DOWN
        elif r > tr:
            return UP
        elif c < tc:
            return RIGHT
        elif c > tc:
            return LEFT
        return STAY


class SequentialVAE(torch.nn.Module):

    def __init__(self, latent_dim: int = 2, permanent: bool = False):
        super().__init__()

        self.latent_dim, self.permanent = latent_dim, permanent
        self.action_encoder = MLP(1, 16, [16])
        self.observation_encoder = MLP(11, 32, [32])
        self.encoder_lstm = torch.nn.GRUCell(32 + 16, 64)
        self.latent_mu = MLP(64, latent_dim, [64])
        self.latent_logvar = MLP(64, latent_dim, [64])
        if not permanent:
            self.encoder_layer = MLP(64, 64, [64])
            self.latent_mu_t = MLP(64, latent_dim, [64])
            self.latent_logvar_t = MLP(64, latent_dim, [64])
            self.latent_layer_t = MLP(latent_dim + 32, 64, [64])
            self.decoder_lstm = torch.nn.GRU(32 + 32, 64, batch_first=True)
        else:
            self.decoder = MLP(32 + 32, 64, [64, 64])
        self.latent_layer = MLP(latent_dim, 32, [32])
        self.action_decoder = MLP(64, 7, [64])

    def forward(self, observations, actions):

        batch, seq, _ = observations.shape

        action_features = self.action_encoder(actions)
        obs_features = self.observation_encoder(observations)

        dec_loss, kl_loss = 0.0, 0.0
        enc_h = torch.zeros(batch, 64, device=actions.device)

        mu_prior = torch.zeros(
            batch,
            self.latent_dim if self.permanent else 2 * self.latent_dim,
            device=actions.device,
        )
        logvar_prior = torch.zeros_like(mu_prior)
        perm_latents, var_latents = [], []

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(
                torch.cat([action_features[:, t], obs_features[:, t]], dim=-1), enc_h
            )
            mu_m = self.latent_mu(enc_h)
            logvar_m = self.latent_logvar(enc_h)
            m = mu_m + torch.randn_like(logvar_m) * torch.exp(0.5 * logvar_m)
            if not self.permanent:
                feat = self.encoder_layer(enc_h)
                mu_mt = self.latent_mu_t(feat)
                logvar_mt = self.latent_logvar_t(feat)
                m_t = mu_mt + torch.randn_like(logvar_mt) * torch.exp(0.5 * logvar_mt)

            dec_features = self.latent_layer(m)
            if not self.permanent:
                dec_h = self.latent_layer_t(torch.cat([dec_features, m_t], dim=-1))
                rec = self.decoder_lstm(
                    torch.cat(
                        [
                            obs_features[:, t:-1],
                            dec_features.unsqueeze(1).repeat_interleave(
                                seq - t - 1, dim=1
                            ),
                        ],
                        dim=-1,
                    ),
                    dec_h.unsqueeze(0),
                )[0]
            else:
                rec = self.decoder(
                    torch.cat(
                        [
                            obs_features[:, t:-1],
                            dec_features.unsqueeze(1).repeat_interleave(
                                seq - t - 1, dim=1
                            ),
                        ],
                        dim=-1,
                    )
                )
            out = self.action_decoder(torch.nn.functional.relu(rec))

            dec_loss += torch.nn.functional.cross_entropy(
                out.flatten(0, 1), actions[:, t + 1 :].flatten().long(), reduction="sum"
            )

            if not self.permanent:
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
            else:
                kl_loss += 0.5 * torch.sum(
                    logvar_prior
                    - logvar_m
                    + (torch.exp(logvar_m) + (mu_m - mu_prior) ** 2)
                    / torch.exp(logvar_prior)
                    - 1,
                    dim=-1,
                ).sum(-1)

            perm_latents.append(m)
            if not self.permanent:
                var_latents.append(m_t)
        perm_latents = torch.stack(perm_latents, dim=1)
        if not self.permanent:
            var_latents = torch.stack(var_latents, dim=1)

        return dec_loss / batch, kl_loss / batch, perm_latents, var_latents


class SequentialVQVAE(torch.nn.Module):
    def __init__(
        self,
        state_size=2,
        dict_size=8,
        emb_dim=1,
        permanent=False,
    ):
        super().__init__()
        self.state_size, self.dict_size, self.emb_dim = state_size, dict_size, emb_dim
        self.permanent = permanent
        self.action_encoder = MLP(1, 16, [16])
        self.observation_encoder = MLP(11, 32, [32])
        self.encoder_lstm = torch.nn.GRUCell(32 + 16, 64)
        self.bottom_encoder = MLP(64, state_size * emb_dim, [64])
        self.bottom_quantizer = EMAMultiCodebookQuantizer(
            state_size, dict_size, emb_dim
        )
        if not permanent:
            self.top_encoder = MLP(state_size * emb_dim, state_size * emb_dim, [64])
            self.top_quantizer = EMAMultiCodebookQuantizer(
                state_size, dict_size, emb_dim
            )
            self.latent_layer_t = MLP(state_size * emb_dim + 32, 64, [64])
            self.decoder_lstm = torch.nn.GRU(32 + 32, 64, batch_first=True)
        else:
            self.decoder = MLP(32 + 32, 64, [64, 64])
        self.latent_layer = MLP(state_size * emb_dim, 32, [32])
        self.action_decoder = MLP(64, 7, [64])

    def forward(self, observations, actions):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        obs_features = self.observation_encoder(observations)

        dec_loss, quantization_loss = 0.0, 0.0
        enc_h = torch.zeros(batch, 64, device=actions.device)
        perm_latents, var_latents = [], []
        perm_ind, var_ind = [], []

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(
                torch.cat([action_features[:, t], obs_features[:, t]], dim=-1), enc_h
            )
            bottom_z = self.bottom_encoder(enc_h).view(
                batch, self.state_size, self.emb_dim
            )
            if not self.permanent:
                top_z = self.top_encoder(bottom_z.view(batch, -1)).view(
                    batch, self.state_size, self.emb_dim
                )
                m_t, loss_t, m_t_ind = self.top_quantizer(top_z)
                quantization_loss += loss_t
            m, loss_b, m_ind = self.bottom_quantizer(bottom_z)
            quantization_loss += loss_b

            dec_features = self.latent_layer(m.view(batch, -1))
            if not self.permanent:
                dec_h = self.latent_layer_t(
                    torch.cat([dec_features, m_t.view(batch, -1)], dim=-1)
                )
                rec = self.decoder_lstm(
                    torch.cat(
                        [
                            obs_features[:, t:-1],
                            dec_features.unsqueeze(1).repeat_interleave(
                                seq - t - 1, dim=1
                            ),
                        ],
                        dim=-1,
                    ),
                    dec_h.unsqueeze(0),
                )[0]
                var_latents.append(m_t.view(batch, -1))
                var_ind.append(m_t_ind)
            else:
                rec = self.decoder(
                    torch.cat(
                        [
                            obs_features[:, t:-1],
                            dec_features.unsqueeze(1).repeat_interleave(
                                seq - t - 1, dim=1
                            ),
                        ],
                        dim=-1,
                    )
                )
            out = self.action_decoder(torch.nn.functional.relu(rec))
            dec_loss += torch.nn.functional.cross_entropy(
                out.flatten(0, 1), actions[:, t + 1 :].flatten().long(), reduction="sum"
            )
            perm_latents.append(m.view(batch, -1))
            perm_ind.append(m_ind)
        perm_latents = torch.stack(perm_latents, dim=1)
        perm_ind = torch.stack(perm_ind, dim=1)
        if not self.permanent:
            var_latents = torch.stack(var_latents, dim=1)
            var_ind = torch.stack(var_ind, dim=1)

        return (
            dec_loss / batch,
            quantization_loss / batch,
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


class GFNCodebook(torch.nn.Module):

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

    def forward(self, indices, training: bool = True):
        batch, state_size = indices.shape
        z_q = torch.gather(
            self.embedding.unsqueeze(0).expand(batch, -1, -1, -1),
            2,
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.emb_dim),
        ).squeeze(2)
        if training:
            one_hot = torch.nn.functional.one_hot(indices, self.dict_size).float()
            cluster_size = one_hot.sum(dim=0)
            dw = torch.einsum("bsk,bsd->skd", one_hot, z_q)
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size + (1 - self.decay) * cluster_size
            )
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
            n = self.ema_cluster_size.sum(dim=-1, keepdim=True)
            cluster_size = (
                (self.ema_cluster_size + self.eps) / (n + self.dict_size * self.eps) * n
            )
            self.embedding.data = self.ema_w / cluster_size.unsqueeze(-1)
        return z_q


class EMGFlowNet:

    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dict_size: int = 4,
        state_size: int = 5,
        permanent: bool = False,
    ):
        if not permanent:
            state_size = 2 * state_size

        self.device = device
        self.n_actions = state_size * dict_size + 1

        self.dict_size = dict_size
        self.state_size = state_size
        self.permanent = permanent

        self.action_encoder = MLP(1, 16, [16]).to(self.device)
        self.observation_encoder = MLP(11, 32, [32]).to(self.device)
        self.encoder_lstm = torch.nn.GRUCell(32 + 16, 64).to(self.device)

        self.pf = MLP(
            input_size=(dict_size + 1) * state_size,
            output_size=64,
            hidden_sizes=[64],
        ).to(self.device)

        self.pf_final = MLP(
            input_size=2 * 64,
            output_size=self.n_actions,
            hidden_sizes=[64],
        ).to(self.device)

        # self.pb = MLP(
        #     input_size=(dict_size + 1) * state_size,
        #     output_size=64,
        #     hidden_sizes=[64],
        # ).to(self.device)

        # self.pb_final = MLP(
        #     input_size=2 * 64,
        #     output_size=self.n_actions - 1,
        #     hidden_sizes=[64],
        # ).to(self.device)

        self.logz = MLP(input_size=64, output_size=1, hidden_sizes=[64]).to(self.device)
        # self.logz = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.gfn_optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.action_encoder.parameters())
                    + list(self.observation_encoder.parameters())
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

        if not permanent:
            self.latent_layer = MLP(int(state_size / 2), 2 * 32, [32]).to(self.device)
            self.latent_layer_t = MLP(int(state_size / 2) + 64, 64, [64]).to(
                self.device
            )
            self.decoder_lstm = torch.nn.GRU(32, 64, batch_first=True).to(self.device)
        else:
            self.latent_layer = MLP(state_size, 32, [64]).to(self.device)
            self.decoder = MLP(32 + 32, 64, [64, 64]).to(self.device)
        self.action_decoder = MLP(64, 7, [64]).to(self.device)
        self.codebook = torch.nn.Parameter(
            torch.randn(state_size, dict_size, 1, device=self.device),
            requires_grad=True,
        ).to(self.device)
        # self.codebook = GFNCodebook(state_size, dict_size, 1).to(self.device)
        # self.codebook = torch.nn.Embedding(dict_size, 1).to(self.device)
        self.d_observation_encoder = MLP(11, 32, [32]).to(self.device)

        self.decoder_optimizer = torch.optim.Adam(
            params=list(self.latent_layer.parameters())
            + list(self.latent_layer_t.parameters() if not self.permanent else [])
            + [self.codebook]
            # + list(self.codebook.parameters())
            + list(
                self.decoder_lstm.parameters()
                if not self.permanent
                else self.decoder.parameters()
            )
            + list(self.d_observation_encoder.parameters())
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

    def train_gfn(self, observations, actions, rand_prob):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        obs_features = self.observation_encoder(observations)
        d_obs_features = self.d_observation_encoder(observations)

        enc_h = torch.zeros(batch, 64, device=actions.device)

        gfn_loss = torch.zeros(batch, seq)

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(
                torch.cat([action_features[:, t], obs_features[:, t]], dim=-1), enc_h
            )

            forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                conditioning=enc_h, rand_prob=rand_prob
            )

            with torch.no_grad():
                z = self.codebook[
                    torch.arange(self.state_size, device=self.device)
                    .unsqueeze(0)
                    .expand(batch, self.state_size),
                    forward_terminal_states.to(self.device),
                ].view(batch, -1)
                # z = self.codebook(forward_terminal_states.to(self.device)).view(batch, -1)
                if not self.permanent:
                    m, m_t = z.chunk(2, dim=-1)
                else:
                    m = z
                dec_features = self.latent_layer(m.view(batch, -1))
                # gamma, beta = dec_features.chunk(2, dim=-1)
                # gamma = 1 + gamma
                # film_out = d_obs_features[:, t:-1] * gamma.unsqueeze(1).repeat_interleave(
                # seq - t - 1, dim=1
                # ) + beta.unsqueeze(1).repeat_interleave(seq - t - 1, dim=1)
                if not self.permanent:
                    dec_h = self.latent_layer_t(
                        torch.cat([dec_features, m_t.view(batch, -1)], dim=-1)
                    )
                    rec = self.decoder_lstm(
                        torch.cat(
                            [
                                d_obs_features[:, t:-1],
                                dec_features.unsqueeze(1).repeat_interleave(
                                    seq - t - 1, dim=1
                                ),
                            ],
                            dim=-1,
                        ),
                        dec_h.unsqueeze(0),
                    )[0]
                else:
                    rec = self.decoder(
                        torch.cat(
                            [
                                d_obs_features[:, t:-1],
                                dec_features.unsqueeze(1).repeat_interleave(
                                    seq - t - 1, dim=1
                                ),
                            ],
                            dim=-1,
                        )
                    )
                    # rec = self.decoder(film_out)
                logprobs = self.action_decoder(torch.nn.functional.relu(rec))

            grewards = (
                -torch.nn.functional.cross_entropy(
                    logprobs.flatten(0, 1),
                    actions[:, t + 1 :].flatten().long(),
                    reduction="none",
                )
                .reshape(batch, seq - t - 1)
                .sum(-1)
                .to(self.device)
                / torch.linspace(99, 1, seq - 1)[t]
                # / torch.logspace(2, -0.5, seq - 1)[t]
            )

            log_z = self.logz(enc_h).squeeze()

            gfn_loss[:, t] = (log_z + forward_log_pf - grewards).pow(2)

        self.gfn_optimizer.zero_grad()
        gfn_loss.mean().backward()
        self.gfn_optimizer.step()

        return gfn_loss.mean().item()

    def train_decoder(self, observations, actions, prob_exponent):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        obs_features = self.observation_encoder(observations)
        d_obs_features = self.d_observation_encoder(observations)

        enc_h = torch.zeros(batch, 64, device=actions.device)

        dec_loss = 0.0

        for t in range(seq - 1):

            with torch.no_grad():

                enc_h = self.encoder_lstm(
                    torch.cat([action_features[:, t], obs_features[:, t]], dim=-1),
                    enc_h,
                )

                forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                    conditioning=enc_h, prob_exponent=prob_exponent
                )

            z = self.codebook[
                torch.arange(self.state_size, device=self.device)
                .unsqueeze(0)
                .expand(batch, self.state_size),
                forward_terminal_states.to(self.device),
            ].view(batch, -1)
            # z = self.codebook(forward_terminal_states.to(self.device)).view(batch, -1)
            if not self.permanent:
                m, m_t = z.chunk(2, dim=-1)
            else:
                m = z
            dec_features = self.latent_layer(m.view(batch, -1))
            # gamma, beta = dec_features.chunk(2, dim=-1)
            # gamma = 1 + gamma
            # film_out = d_obs_features[:, t:-1] * gamma.unsqueeze(1).repeat_interleave(
            # seq - t - 1, dim=1
            # ) + beta.unsqueeze(1).repeat_interleave(seq - t - 1, dim=1)
            if not self.permanent:
                dec_h = self.latent_layer_t(
                    torch.cat([dec_features, m_t.view(batch, -1)], dim=-1)
                )
                rec = self.decoder_lstm(
                    torch.cat(
                        [
                            d_obs_features[:, t:-1],
                            dec_features.unsqueeze(1).repeat_interleave(
                                seq - t - 1, dim=1
                            ),
                        ],
                        dim=-1,
                    ),
                    dec_h.unsqueeze(0),
                )[0]
            else:
                rec = self.decoder(
                    torch.cat(
                        [
                            d_obs_features[:, t:-1],
                            dec_features.unsqueeze(1).repeat_interleave(
                                seq - t - 1, dim=1
                            ),
                        ],
                        dim=-1,
                    )
                )
                # rec = self.decoder(film_out)
            logprobs = self.action_decoder(torch.nn.functional.relu(rec))

            dec_loss += (
                torch.nn.functional.cross_entropy(
                    logprobs.flatten(0, 1),
                    actions[:, t + 1 :].flatten().long(),
                    reduction="sum",
                )
                / batch
            )

        self.decoder_optimizer.zero_grad()
        dec_loss.backward()
        self.decoder_optimizer.step()

        return dec_loss.mean().item()

    def sample_latents(self, observations, actions, rand_prob, prob_exponent):

        batch, seq, _ = actions.shape

        action_features = self.action_encoder(actions)
        obs_features = self.observation_encoder(observations)
        latents, indices = [], []

        enc_h = torch.zeros(batch, 64, device=actions.device)

        for t in range(seq - 1):

            enc_h = self.encoder_lstm(
                torch.cat([action_features[:, t], obs_features[:, t]], dim=-1), enc_h
            )

            forward_terminal_states, forward_log_pf = self.sample_ar_trajectories(
                conditioning=enc_h, rand_prob=rand_prob, prob_exponent=prob_exponent
            )
            z = self.codebook[
                torch.arange(self.state_size, device=self.device)
                .unsqueeze(0)
                .expand(batch, self.state_size),
                forward_terminal_states.to(self.device),
            ].view(batch, -1)
            # z = self.codebook(forward_terminal_states.to(self.device)).view(batch, -1)
            latents.append(z)
            indices.append(forward_terminal_states)
        latents = torch.stack(latents, dim=1)
        indices = torch.stack(indices, dim=1)
        return latents, indices
