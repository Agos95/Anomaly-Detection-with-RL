# %%
from torch import nn
import torch.nn.functional as F

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


class DQNSolver(object):
    def __init__(self,
                 n_episodes=1000,
                 n_win_ticks=195,
                 max_env_steps=None,
                 gamma=1.0,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_log_decay=0.995,
                 alpha=0.01,
                 alpha_decay=0.01,
                 batch_size=64,
                 monitor=False,
                 quiet=False):
        pass


# %%
