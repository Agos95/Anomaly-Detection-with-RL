# %%[markdown]

# [DPLAN implementation](https://github.com/lflfdxfn/DPLAN-Implementation)
# Paper: Toward Deep Supervised Anomaly Detection: Reinforcement Learning from Partially Labeled Anomaly Data

# see also: https://medium.com/analytics-vidhya/introduction-to-reinforcement-learning-rl-in-pytorch-c0862989cc0e

# %%
import json
import numpy as np
from sklearn.metrics import confusion_matrix

import gym
from gym.spaces import Discrete, Box, MultiBinary

# %%


class AnomalyDetectionEnv(gym.Env):

    def __init__(self, X, y, w_tp=5, w_tn=5, w_fp=-1, w_fn=-1, debug=False) -> None:
        """
        Custom environment for Anomaly Detection

        Parameters
        ----------
        X : array-like of shape `(n, n_timesteps, n_features)`
            Array of input data with:
            - `n`: number of different samples (timeseries) in the daatset
            - `n_timesteps`: number of timestamps in each timeseries
            - `n_features`: number of features in each timeseries
        y : array-like of shape `(n,)` or `(n, n_timesteps)`
            Label associated to each sample (0:normal, 1:anomaly).
            If shape is `(n,)`, each timeseries is assigned a label looking only at the last timestep.
            If shape is `(n, n_timesteps)`, each point in the timeseries is labelled.
        """
        super().__init__()
        self.debug = debug

        # dataset
        self.X, self.y = X, y
        self.n, self.n_timesteps, self.n_features = self.X.shape
        self.label_type = "last" if len(self.y.shape) == 1 else "all"

        # Action Space: {0, 1} = {normal, anomaly}
        self.action_space = Discrete(
            2) if self.label_type == "last" else MultiBinary(self.n_timesteps)
        # Observation Space: each observation is a time series
        self.observation_space = Box(
            low=0, high=1, shape=(self.n_timesteps, self.n_features), dtype=self.X.dtype)

        # weights for reward
        self.reward_weights = np.array([w_tn, w_fp, w_fn, w_tp])

        # initial state
        self.state_index = 0
        self.state = self.X[self.state_index]
        self.t = 0

    def reset(self):
        """
        Reset the environment for next episode.

        Returns
        -------
        state : array
            First state of the env.
        """
        self._print("Resetting the environment.")
        self.state_index = 0
        self.state = self.X[self.state_index]
        self.t = 0
        return self.state

    def step(self, action):
        """
        Parameters
        ----------
        action : int
            Action chosen by the agent.

        Returns
        -------
        next_state : int
            Next state from `self.observation_space`, which corresponds to an index of `self.observations`.
        reward : float
            Reward returned as result for the chosen action.
        done : bool
            True if the episode has ended. Always False in this implementation.
        info : dict
            Additional information.
        """

        # store current state
        current_state_index = self.state_index
        current_state = self.state

        # ########## #
        # GET REWARD #
        # ########## #

        try:
            _ = len(action)
            true_action = self.y[current_state_index]
        except:
            action = int(action)
            action = [action]
            true_action = [self.y[current_state_index]]

        cm = confusion_matrix(true_action, action, labels=[0, 1])

        reward = self.reward_weights * cm.ravel()
        reward = np.sum(reward)

        # ####################### #
        # MOVE TO NEXT STATE s_t1 #
        # ####################### #

        # just go to next state without any particular logic
        next_state_index = (current_state_index + 1)
        try:
            next_state = self.X[next_state_index]
        except:
            next_state = None

        # move to next state
        self.state_index = next_state_index
        self.state = next_state
        done = (self.state_index == self.n)

        # info
        info = {
            "t": self.t,
            "state": current_state,
            "action": action,
            "true_action": true_action,
            "reward": reward,
            "state": next_state,
            "done": done
        }
        self.t += 1

        return next_state, reward, done, info

    def __repr__(self):
        msg = "\n".join(
            f"AnomalyDetection Environment",
            f"----------------------------",
            f"X shape = {self.X.shape}",
            f"y shape = {self.y.shape}",
            f"Action Space      = {self.action_space}",
            f"Observation Space = {self.observation_space}",
            f"Reward weights (tn, fp, fn, tp) = {self.reward_weights}"
        )
        return msg

    def _print(self, msg):
        if self.debug:
            print(msg)
