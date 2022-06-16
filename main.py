# %%
import argparse
from os.path import join
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch

from dataset import TSDataset
from env import AnomalyDetectionEnv
from agent import DQN
from policy import EpsilonGreedyPolicy
from memory import ReplayMemory


# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/config.json", help="Configuration file")
    args = parser.parse_args()
    return vars(args)


def main(args):

    with open(args["config"], "r") as f:
        cfg = json.load(f)

    # load and prepare dataset
    data = TSDataset(join("data", "rastro.tar.gz"), **cfg["data"])
    scaler = MinMaxScaler() if cfg["train_test"].pop("scaler") else None
    X_train, X_test, y_train, y_test = data.train_test_split(
        **cfg["train_test"])

    # environment
    env = AnomalyDetectionEnv(X_train, y_train, **cfg["env"])

    # agent & dqn stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 2 if cfg["data"]["label_type"] == "last" else X_train.shape[1]
    policy_net = DQN(n_features=X_train.shape[2],
                     output_dim=output_dim, **cfg["agent"])
    target_net = DQN(n_features=X_train.shape[2],
                     output_dim=output_dim, **cfg["agent"])
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), **cfg["optimizer"])
    # softmax + BCELoss (since our network does not have softmax layer at the end)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
