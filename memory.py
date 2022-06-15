# %%
from collections import deque
import numpy as np

# %%


class ReplayMemory(object):

    def __init__(self, capacity, seed=None):
        """
        Replay Memory

        Parameters
        ----------
        capacity : int
            Number of instances to keep in memory.
        seed : int, optional
            Seed to initialize the RNG which extracts the batches.
        """

        self.memory = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, state, action, next_state, reward):
        """
        Add the tuple `(state, action, next_state, reward)` to the replay memory.
        """
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        """
        Get a batch from the replay memory.
        """
        batch_size = min(batch_size, len(self))
        return self.rng.choice(self.memory, size=batch_size, replace=False)

    def __len__(self):
        return len(self.memory)
