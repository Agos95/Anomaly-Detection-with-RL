# %%
import numpy as np

# %%


class EpsilonGreedyPolicy(object):
    """
    EpsilonGreedyPolicy implementation:
    - with `p=epsilon` choose random action
    - with `p=1-epsilon` choose the best action

    see: https://github.com/keras-rl/keras-rl/blob/216c3145f3dc4d17877be26ca2185ce7db462bad/rl/policy.py#L141

    Parameters
    ----------
    epsilon : float [default=.1]
        Threshold to choose random action instead of best one.
    seed : int, optional
        Seed to initialize the RNG to sample actions.
    """

    def __init__(self, epsilon=.1, seed=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = len(q_values)

        if self.rng.random() < self.eps:
            action = self.rng.integers(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action
