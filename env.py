# %%[markdown]

# [DPLAN implementation](https://github.com/lflfdxfn/DPLAN-Implementation)
# Paper: Toward Deep Supervised Anomaly Detection: Reinforcement Learning from Partially Labeled Anomaly Data

# see also: https://medium.com/analytics-vidhya/introduction-to-reinforcement-learning-rl-in-pytorch-c0862989cc0e

# %%
import numpy as np
import pandas as pd

import gym
from gym.spaces import Discrete

# %%


class AnomalyDetectionEnv(gym.Env):

    def __init__(self, df, anomaly, window="1H", stride="10min", columns=None) -> None:
        """
        Custom environment for Anomaly Detection

        Parameters
        ----------
        df : pandas Dataframe
            Dataframe with the time series as columns.
        anomaly : list-like, optional
            List of Timestamps which corresponds to anomalies.
        columns : list-like, optional
            List of columns of `df` to use. Default is `None`, which means all the columns.
        window : str [default="1H"]
            Size of the temporal window. See pandas documentation for offsets strings.
        stride : str [default="10min"]
            Stride between two consecutive temporal windows. See pandas documentation for offsets strings.
        """
        super().__init__()

        # dataset
        self.anomaly = anomaly
        self.window = pd.Timedelta(window)
        self.stride = pd.Timedelta(stride)
        self.columns = df.columns if (columns is None or not set(
            columns).issubset(df.columns)) else columns
        self.df = df[self.columns]

        self.observations, self.labels = self._make_sliding_windows(
            self.df, self.window, self.stride, self.anomaly)
        self.n = len(self.observations)
        self.ts_length, self.ts_features = self.observations[0].shape

        # Action Space: {0, 1} = {normal, anomaly}
        self.action_space = Discrete(2)
        # Observation Space: each observation is a time series
        self.observation_space = Discrete(self.n)

        # initial state
        self.state = None

    def _make_sliding_windows(df, window, stride, anomaly=None):
        """
        Create sliding windows from a DataFrame and optionally assigns a label based on `anomaly`.
        The label is given looking at the last timestamp of each time series.

        Parameters
        ----------
        df : pandas DataFrame
            Dataset with time index.
        window : pandas Timedelta
            Size of the temporal window.
        stride : pandas Timedelta
            Stride between consecutive temporal windows.
        anomaly : list-like, optional
            List of Timestamps corresponding to anomalies.

        Returns
        -------
        dfs : list of pandas DataFrames
            Each DataFrame corresponds to a time window of length `window`.
            Consecutive DataFrames are separated by `stride`.

        y : list of int
            List of labels corresponding to each time window.
            A DataFrame is labelled 1 (anomaly) or 0 (normal)
            if the last timestamp is in `anomaly` or not.
            It is empty if `anomaly` is `None`.

        """

        start = df.index.min()
        end = df.index.max()

        t0 = start
        t1 = start + window

        dfs, y = [], []
        while t1 < end:
            dfs.append(df.loc[t0:t1, :])
            t0 = t0 + stride
            t1 = t0 + window

        if anomaly is not None:
            for df in dfs:
                last = df.index.max()
                an = 1 if last in anomaly else 0
                y.append(an)

        return dfs, y

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = None
        return

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
        current_state = self.state

        # ####################### #
        # MOVE TO NEXT STATE s_t1 #
        # ####################### #

        # just go to next state without any particular logic
        next_state = current_state + 1

        # ########## #
        # GET REWARD #
        # ########## #

        reward =

        # move to next state
        self.state = next_state

        done = (self.state == (self.n - 1))

        # info
        info = {
            "State (t)": current_state,
            "Action (t)": action,
            "State (t+1)": next_state,
        }

        return next_state, reward, done, info
