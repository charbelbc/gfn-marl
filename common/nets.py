import torch
import re
import numpy as np


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Vocabulary:

    def __init__(self):
        self.max_size = 100
        self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


class InstructionsPreprocessor(object):

    def __init__(self):
        self.vocab = Vocabulary()

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0
        for obs in obss:
            tokens = re.findall("([a-z]+)", obs.lower())
            instr = np.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = np.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, : len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


class ImageBOWEmbedding(torch.nn.Module):

    def __init__(self, max_value, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(3 * max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(
            inputs.device
        )
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class FiLM(torch.nn.Module):

    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=imm_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = torch.nn.BatchNorm2d(imm_channels)
        self.conv2 = torch.nn.Conv2d(
            in_channels=imm_channels,
            out_channels=out_features,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = torch.nn.BatchNorm2d(out_features)

        self.weight = torch.nn.Linear(in_features, out_features)
        self.bias = torch.nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return torch.nn.functional.relu(self.bn2(out))


class ACNetwork(torch.nn.Module):

    def __init__(self, memory_size: int = 512, action_dim: int = 5, n_agents: int = 2):
        super().__init__()
        self.memory_size = memory_size
        self.action_dim = action_dim

        self.image_emb = ImageBOWEmbedding(147, 128)
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )

        self.film_pool = torch.nn.MaxPool2d(kernel_size=(7, 7), stride=2)

        self.instr_emb = torch.nn.Embedding(100, 128)
        self.instr_rnn = torch.nn.GRU(128, 128, batch_first=True)

        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            mod = FiLM(
                in_features=128,
                out_features=128,
                in_channels=128,
                imm_channels=128,
            )
            self.controllers.append(mod)
            self.add_module("FiLM_" + str(ni), mod)

        self.memory_rnn = torch.nn.LSTMCell(128, memory_size)

        self.agent_id = torch.nn.Embedding(n_agents, 16)

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(memory_size + 16, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 64),
            torch.nn.Linear(64, action_dim),
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 64),
            torch.nn.Linear(64, 1),
        )

        self.apply(initialize_parameters)

    def forward(self, observations, memory, instr_tensor):

        batch, agents, *channels = observations.shape

        lengths = (instr_tensor != 0).sum(1).long()
        out, _ = self.instr_rnn(self.instr_emb(instr_tensor))
        instr_embedding = out[range(len(lengths)), lengths - 1, :]

        features = self.convnet(
            self.image_emb(
                observations.transpose(0, 1).reshape(batch * agents, *channels)
            )
        )

        for controller in self.controllers:
            x = controller(features, instr_embedding.repeat(agents, 1)) + features
            features = x

        x = torch.nn.functional.relu(self.film_pool(features)).flatten(start_dim=1)

        hidden = self.memory_rnn(
            x,
            (
                memory.transpose(0, 1).flatten(0, 1)[:, : self.memory_size],
                memory.transpose(0, 1).flatten(0, 1)[:, self.memory_size :],
            ),
        )
        embedding = hidden[0]
        actor_logits = self.actor(
            torch.cat(
                [
                    # torch.nn.functional.one_hot(
                    #     torch.arange(agents).repeat_interleave(batch),
                    #     num_classes=agents,
                    # )
                    # .float()
                    # .to(observations.device),
                    self.agent_id(
                        torch.arange(agents)
                        .repeat_interleave(batch)
                        .to(observations.device)
                    ),
                    embedding,
                ],
                dim=-1,
            )
        ).log_softmax(dim=-1)
        value = self.critic(embedding[:batch])

        return (
            actor_logits.reshape(agents, batch, self.action_dim).transpose(0, 1),
            value,
            torch.cat(hidden, dim=1)
            .reshape(agents, batch, 2 * self.memory_size)
            .transpose(0, 1),
        )


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            torch.nn.init.constant_(param, 0)
        elif "weight" in name:
            torch.nn.init.orthogonal_(param, gain=gain)


class MPE_ACNetwork(torch.nn.Module):

    def __init__(self, action_dim: int = 5, n_agents: int = 2):
        super().__init__()
        self.action_dim = action_dim

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(6 * n_agents, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(6 * n_agents * n_agents, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        orthogonal_init(self.actor)
        orthogonal_init(self.critic)

    def forward(self, observations):

        actor_logits = self.actor(observations)  # .log_softmax(dim=-1)
        value = self.critic(observations.flatten(1))

        return actor_logits, value


class MPE_Actor(torch.nn.Module):

    def __init__(self, action_dim: int = 5, n_agents: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(6 * n_agents, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, action_dim),
        )
        orthogonal_init(self.actor[0])
        orthogonal_init(self.actor[2])
        orthogonal_init(self.actor[4])
        orthogonal_init(self.actor[6], gain=0.01)

    def forward(self, observations):
        actor_logits = self.actor(observations)
        return actor_logits.log_softmax(-1)


class MPE_Critic(torch.nn.Module):

    def __init__(self, action_dim: int = 5, n_agents: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(6 * n_agents * n_agents, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )
        orthogonal_init(self.critic)

    def forward(self, observations):
        value = self.critic(observations)
        return value


class MPE_RNN_Actor(torch.nn.Module):

    def __init__(self, action_dim: int = 5, n_agents: int = 2):
        super().__init__()
        self.action_dim = action_dim

        self.actor_1 = torch.nn.Linear(6 * n_agents, 64)
        self.actor_2 = torch.nn.Linear(64, 64)
        self.actor_rnn = torch.nn.GRUCell(64, 64)
        self.actor_3 = torch.nn.Linear(64, action_dim)
        self.activation = torch.nn.Tanh()
        self.actor_rnn_hidden = None

        orthogonal_init(self.actor_1)
        orthogonal_init(self.actor_2)
        orthogonal_init(self.actor_rnn)
        orthogonal_init(self.actor_3, gain=0.01)

    def forward(self, observations):

        batch, agents, features = observations.shape
        x = self.activation(self.actor_1(observations.flatten(0, 1)))
        x = self.activation(self.actor_2(x))
        self.actor_rnn_hidden = self.actor_rnn(x)
        actor_logits = self.actor_3(self.actor_rnn_hidden)

        return actor_logits.reshape(batch, agents, -1).log_softmax(-1)


class MPE_RNN_Critic(torch.nn.Module):

    def __init__(self, action_dim: int = 5, n_agents: int = 2):
        super().__init__()
        self.action_dim = action_dim

        self.critic_1 = torch.nn.Linear(6 * n_agents * n_agents, 64)
        self.critic_2 = torch.nn.Linear(64, 64)
        self.critic_rnn = torch.nn.GRUCell(64, 64)
        self.critic_3 = torch.nn.Linear(64, 1)
        self.activation = torch.nn.Tanh()
        self.critic_rnn_hidden = None

        orthogonal_init(self.critic_1)
        orthogonal_init(self.critic_2)
        orthogonal_init(self.critic_rnn)
        orthogonal_init(self.critic_3)

    def forward(self, observations):

        batch, agents, features = observations.shape
        y = self.activation(self.critic_1(observations.flatten(0, 1)))
        y = self.activation(self.critic_2(y))
        self.critic_rnn_hidden = self.critic_rnn(y)
        value = self.critic_3(self.critic_rnn_hidden)

        return value.reshape(batch, agents, -1)
