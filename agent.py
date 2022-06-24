# %%
from copy import deepcopy
from tqdm import trange
import logging

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from gym import Env

from memory import ReplayMemory
from policy import EpsilonGreedyPolicy

# %%


class DQN(nn.Module):

    def __init__(self,
                 n_features,
                 hidden_lstm_dim,
                 linear_dim,
                 output_dim,
                 dropout=.1,
                 lstm_kwargs={
                     "num_layers": 1,
                     "dropout": 0,
                     "_bidirectional": False}
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
                `"num_layers": 1`,
                `"dropout": 0`,
                `"bidirectional": False}`
            }
        """

        super().__init__()
        self.n_features = n_features
        if isinstance(linear_dim, int):
            linear_dim = [linear_dim]
        D = 1
        if "lstm_bidirectional" in lstm_kwargs and lstm_kwargs["lstm_bidirectional"] is True:
            D = 2
        #linear_dim = [D * hidden_lstm_dim] + linear_dim

        # Layers
        self.lstm_layers = nn.LSTM(
            input_size=n_features, hidden_size=hidden_lstm_dim,
            batch_first=True,
            **lstm_kwargs)
        self.linear = nn.Sequential(
            nn.Flatten(),
            *[nn.LazyLinear(linear_dim[i]) for i in range(len(linear_dim))])
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

    def __init__(self, env: Env, model, loss_fn, model_kwargs={}, optimizer=Adam, optimizer_kwargs={}, batch_size=16, replay_memory_capacity=1000, gamma=.9, min_sample_for_training=64, device="auto", episodes=1000, steps=None, policy_kwargs={}, target_net_update=25):
        """
        env : OpenAI Gym Env
            Environment.
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
        gamma : float, default .9
            Discount factor.
        min_sample_for_training : int, default 64
            Minimum number of samples in the replay memory
            to start training the network.
        device : str, default "auto"
            Device to use for torch models.
            Default is "auto", which means "gpu" if available or "cpu" otherwise.
        episodes : int, default 1000
            Number of episodes.
        steps : int, optional
            Numbr of steps per episode.
            Default is None, which corresponds to the number of examples in the env.
        policy_kwargs : dict, default {}
            Additional arguments to be passed to the Epsilon Greedy Policy
        target_net_update : int, default 25
            Number of episodes between updates of the target network.
        """

        # gym env
        self.env = env
        self.episodes = episodes
        self.steps = steps if steps is not None else self.env.n

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # torch related things
        self.policy_net: nn.Module = model(**model_kwargs)
        self.target_net: nn.Module = model(**model_kwargs)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer: torch.optim.Optimizer = optimizer(
            self.policy_net.parameters(), **optimizer_kwargs)
        self.loss_fn = loss_fn
        # move everything to correct device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.loss_fn.to(self.device)

        # replay memory
        self.memory = ReplayMemory(replay_memory_capacity)
        self.batch_size = batch_size
        self.min_sample_for_training = min_sample_for_training
        # discount factor
        self.gamma = gamma
        # policy
        self.policy = EpsilonGreedyPolicy(**policy_kwargs)
        # target net update
        self.target_net_update = target_net_update

        # rewards and metrics
        self.rewards = np.full((self.episodes, self.steps), np.nan)
        self.ep_rewards = np.full(self.episodes, np.nan)
        self.step_rewards = np.full(self.steps, np.nan)

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
        state_action_values = q_values.max(dim=1)[0].detach()

        # compute value function with target net
        with torch.no_grad():
            self.target_net.eval()
            q_values_target = self.target_net(states)

        next_state_max_q_values = q_values_target.max(dim=1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = rewards + \
            (next_state_max_q_values * self.gamma)

        # compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self):
        for e in trange(self.episodes, desc="Episode"):
            self.env.reset()
            state = self.env.state
            done = False
            score = 0

            for s in trange(self.steps, leave=False, desc="Step"):
                # get q-values from target net
                with torch.no_grad():
                    q_values = self.policy_net(
                        torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device))
                # select next action using the policy
                action = self.policy.select_action(q_values.squeeze())
                # do one step in the env given the chosen action
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.rewards[e, s] = reward
                # update replay memory
                self.memory.push(state, action, next_state, reward)
                # update the network
                if len(self.memory) >= self.min_sample_for_training:
                    self.update()

                if done:
                    break

                # update the state
                state = next_state

            if (e + 1) % self.target_net_update:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.ep_rewards[e] = score
            logging.info(f"Episode {e+1:5d} - Score {score:5d}")

        self.step_rewards = np.nanmean(self.rewards, axis=1)

        # %%
