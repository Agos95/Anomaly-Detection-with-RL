# %%
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout

from rl.core import Processor

# %%


def kDQN(input_shape, lstm_units=25, lstm_kwargs={}, dropout=.1, linear_units=25, linear_kwargs={"activation": "relu"}, output_dim=2):
    """
    Deep Q Network.

    Parameters
    ----------
    input_shape : tuple
        Dimension of the input.
    lstm_units : int or list of int, default 25
        Number of units of lstm layers.
    lstm_kwargs : dict, default {}
        Additional arguments for LSTM layers.
    dropout : floar, default .1
        Droput rate between recurrent and linear part of the Network.
    lstm_units : int or list of int, default 25
        Number of units of linear layers.
    linear_kwargs : dict, default {"activation":"relu"}
        Additional arguments for Linear layers.
    output_dim : int, default 2
        Number of output units.
    """

    if isinstance(lstm_units, int):
        lstm_units = [lstm_units]
    if isinstance(linear_units, int):
        linear_units = [linear_units]

    inputs = Input(input_shape)
    # Recurrent part
    if len(lstm_units) == 1:
        x = LSTM(lstm_units[0], return_sequences=False, **lstm_kwargs)(inputs)
    else:
        x = LSTM(lstm_units[0], return_sequences=True, **lstm_kwargs)(inputs)
        for l in lstm_units[1:-1]:
            x = LSTM(l, return_sequences=True, **lstm_kwargs)(x)
        x = LSTM(lstm_units[-1], return_sequences=False, **lstm_kwargs)(x)
    # Dropout
    x = Dropout(dropout)(x)
    # Linear part
    for l in linear_units:
        x = Dense(l, **linear_kwargs)(x)
    x = Dropout(dropout)(x)
    # output
    outputs = Dense(output_dim, activation="softmax")(x)

    return Model(inputs, outputs)


# %%

class CustomProcessor(Processor):
    def process_state_batch(self, batch):
        # see: https://github.com/keras-rl/keras-rl/issues/113#issuecomment-482913631

        return np.squeeze(batch, axis=1)
