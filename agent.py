# %%
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from memory import ReplayMemory

# %%


class DQN(nn.Module):

    def __init__(self,
                 n_features,
                 hidden_lstm_dim,
                 linear_dim,
                 output_dim,
                 dropout=.1,
                 lstm_kwargs={
                     "num_lstm_layers": 1,
                     "lstm_dropout": 0,
                     "lstm_bidirectional": False}
                 ):
        """
        Deep Q Network.

        Parameters
        ----------
        n_features : int
            Number of features in the dataset.
        hidden_lstm_dim : int
            Dimension of the LSTM hidden state.
        linear_dim : list of int or int
            Dimensions of the linear layers (Ouput excluded).
        output_dim : int
            Dimension of the output.
        dropout : float [default=.1]
            Dropout rate after the linear layers before the output.
        lstm_kwargs : dict, optional
            Additional arguments to be passed to `nn.LSTM` layer.
            Defaults to:
            {
                `"num_lstm_layers": 1`,
                `"lstm_dropout": 0`,
                `"lstm_bidirectional": False}`
            }
        """

        super().__init__()
        self.n_features = n_features
        if isinstance(linear_dim, int):
            linear_dim = [linear_dim]
        D = 1
        if "lstm_bidirectional" in lstm_kwargs and lstm_kwargs["lstm_bidirectional"] is True:
            D = 2
        linear_dim = [D * hidden_lstm_dim] + linear_dim

        # Layers
        self.lstm_layers = nn.LSTM(
            input_size=n_features, hidden_size=hidden_lstm_dim, **lstm_kwargs)
        self.linear = nn.ModuleList(
            [nn.Linear(linear_dim[i], linear_dim[i + 1]) for i in range(len(linear_dim) - 1)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(linear_dim[-1], output_dim)

    def forward(self, x):
        x, _ = self.lstm_layers(x)
        x = F.relu(x)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

# %%


class DQNAgent(object):

    def __init__(self, model, loss_fn, model_kwargs={}, optimizer: torch.optim.Optimizer = Adam, optimizer_kwargs={}, batch_size=16, replay_memory_capacity=1000, gamma=.99, min_sample_for_training=64, device="auto"):
        """
        model : torch.nn.Module
            Policy Network.
        model_kwargs : dict, default {}
            Additional arguments for the policy_net.
        loss_fn : torch loss function
            Loss function to use to train the network.
        optimizer : torch.optim.Optimizer, default Adam
            Optimizer to use during training.
        optmizer_kwargs : dict, default {}
            Additional arguments for the optimizer.
        replay_memory_capacity : int, default 1000
            Replay memory capacity.
        gamma : float, default .99
            Discount factor.
        min_sample_for_training : int, default 64
            Minimum number of samples in the replay memory
            to start training the network.
        device : str, default "auto"
            Device to use for torch models.
            Default is "auto", which means "gpu" if available or "cpu" otherwise.
        """

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # torch related things
        self.policy_net: nn.Module = model(**model_kwargs)
        self.target_net: nn.Module = model(**model_kwargs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer: torch.optim.optimizer = optimizer(
            self.policy_net.parameters(), **optimizer_kwargs)
        self.loss_fn = loss_fn
        # move everything to correct device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer.to(self.device)
        self.loss_fn.to(self.device)

        # replay memory
        self.memory = ReplayMemory(replay_memory_capacity)
        self.batch_size = batch_size
        # discount factor
        self.gamma = gamma
        # others
        self.min_sample_for_training = min_sample_for_training

    def update(self):
        batch = self.memory.sample(self.batch_size)
        # account for case when number of samples in memory are less than batch size
        batch_size = len(batch)
        states = torch.tensor([b[0] for b in batch],
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch],
                               dtype=torch.int32, device=self.device)
        rewards = torch.tensor([b[3] for b in batch],
                               dtype=torch.float32, device=self.device)

        # Compute a mask of non-final states (all the elements where the next state is not None)
        # non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32, device=device) # the next state can be None if the game has ended
        #non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

        # Compute Q values
        self.policy_net.train()
        q_values = self.policy_net(states)

# %%
