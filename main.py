# %%
from os.path import join
import json
import pandas as pd

import torch

from env import AnomalyDetectionEnv
from agent import DQN
from policy import EpsilonGreedyPolicy
from memory import ReplayMemory


# %%


def main():
    # read data
    df = pd.read_csv(join("data", "rastro_1min.tar.gz"),
                     parse_dates=[0], index_col=0)
    with open(join("config", "config.json"), "r") as f:
        config = json.load(f)
    # known anomalies
    anomaly = pd.date_range(start="2016/07/03 09:00:00",
                            end="2016/07/03 16:00:00", freq="1min").tolist() + \
        pd.date_range(start="2016/07/10 09:00:00",
                      end="2016/07/10 16:00:00", freq="1min").tolist() + \
        pd.date_range(start="2016/07/17 09:00:00",
                      end="2016/07/17 16:00:00", freq="1min").tolist() + \
        pd.date_range(start="2016/07/24 09:00:00",
                      end="2016/07/24 16:00:00", freq="1min").tolist()

    # create custom gym Env
    env = AnomalyDetectionEnv(
        df=df, anomaly=anomaly, window=config["window"], stride=config["stride"], columns=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(n_features=df.shape[1], hidden_lstm_dim=config["hidden_lstm_dim"], linear_dim=config["linear_dim"],
                output_dim=2, dropout=config["dropout"], lstm_kwargs=config["lstm_kwargs"])


# %%
if __name__ == "__main__":
    main()
