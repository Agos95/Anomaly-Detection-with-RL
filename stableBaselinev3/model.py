# %%
import gym
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# %%


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 50, lstm_dim: int = 25, dropout: float = .1):
        """
        Feature Extractore for baselines Deep Q Network.

        Parameters
        ----------
        observation_space : gym.Space
            OpenAI Gym observation space.
        feature_dim : int, default 50
            Dimension of the feature space.
        lstm_dim : int, default 25
            Dimension of the LSTM hidden state.
        droput : float, default .1
            Dropout rate between lstm and linear layer.
        """
        super().__init__(observation_space, features_dim)

        seq_length, num_features = observation_space.shape
        linear_in_features = seq_length * lstm_dim

        self.lstm = nn.LSTM(num_features, lstm_dim,
                            num_layers=1, batch_first=True)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(linear_in_features, features_dim)

    def forward(self, x):
        #x = self.net(x)
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.linear(x)
        x = F.relu(x)
        return x

# %%

# class FeatureExtractor(BaseFeaturesExtractor):

#     def __init__(self,
#                  n_features,
#                  hidden_lstm_dim,
#                  linear_dim,
#                  dropout=.1,
#                  lstm_kwargs={
#                      "num_layers": 1,
#                      "dropout": 0,
#                      "bidirectional": False}
#                  ):
#         """
#         Feature extractor for baselines Deep Q Network.

#         Parameters
#         ----------
#         n_features : int
#             Number of features in the dataset.
#         hidden_lstm_dim : int
#             Dimension of the LSTM hidden state.
#         linear_dim : list of int or int
#             Dimensions of the linear layers (Ouput excluded).
#         dropout : float [default=.1]
#             Dropout rate after the linear layers before the output.
#         lstm_kwargs : dict, optional
#             Additional arguments to be passed to `nn.LSTM` layer.
#             Defaults to:
#             {
#                 `"num_layers": 1`,
#                 `"dropout": 0`,
#                 `"bidirectional": False}`
#             }
#         """

#         super().__init__()
#         self.n_features = n_features
#         if isinstance(linear_dim, int):
#             linear_dim = [linear_dim]
#         D = 1
#         if "lstm_bidirectional" in lstm_kwargs and lstm_kwargs["lstm_bidirectional"] is True:
#             D = 2
#         #linear_dim = [D * hidden_lstm_dim] + linear_dim

#         # Layers
#         self.lstm_layers = nn.LSTM(
#             input_size=n_features, hidden_size=hidden_lstm_dim,
#             batch_first=True,
#             **lstm_kwargs)
#         self.linear = nn.Sequential(
#             nn.Flatten(),
#             *[nn.LazyLinear(linear_dim[i]) for i in range(len(linear_dim))])
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x, _ = self.lstm_layers(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.linear(x)
#         x = F.relu(x)
#         return x
