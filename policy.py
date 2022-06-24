# %%
import numpy as np

# %%


class EpsilonGreedyPolicy(object):

    def __init__(self, epsilon=1., min_epsilon=.05, decay=.995, seed=None):
        """
        EpsilonGreedyPolicy implementation:
        - with `p=epsilon` choose random action
        - with `p=1-epsilon` choose the best action

        see: https://github.com/keras-rl/keras-rl/blob/216c3145f3dc4d17877be26ca2185ce7db462bad/rl/policy.py#L141

        Parameters
        ----------
        epsilon : float, default=1.
            Threshold to choose random action instead of best one.
        epsilon_min : float, default .05
            Minimum value for epsilon.
        decay : float, default .995
            Decay per step.
        seed : int, optional
            Seed to initialize the RNG to sample actions.
        """
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        self.rng = np.random.default_rng(seed=seed)

    def _update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def select_action(self, q_values):
        """
        Select next action.

        Parameters
        ----------
        q_values : array-like with dimension 1
            Array of q-values.

        Returns
        -------
        action : int
            Next action to be performed.
        """
        assert q_values.ndim == 1, f"Expected dim=1, found {q_values.ndim}"
        nb_actions = len(q_values)

        if self.rng.random() < self.epsilon:
            action = self.rng.integers(0, nb_actions)
        else:
            action = np.argmax(q_values)
        self._update_epsilon()

        return action
