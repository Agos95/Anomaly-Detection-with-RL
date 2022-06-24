# %%
import argparse
import os
from os.path import join, splitext, basename
import json
import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.optim import Adam

from dataset import TSDataset
from env import AnomalyDetectionEnv
from agent import DQN, DQNSolver


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

    logger_name = join("logs", basename(splitext(args["config"])[0]))
    logging.basicConfig(
        filename=logger_name, filemode="w", format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO)
    logging.info(
        f"Configuration file '{args['config']}':\n" + json.dumps(cfg, indent=4))

    # load and prepare dataset
    logging.info("Prepraing the dataset...")
    data = TSDataset(join("data", "rastro.tar.gz"), **cfg["data"])
    scaler = MinMaxScaler() if cfg["train_test"].pop("scaler") else None
    X_train, X_test, y_train, y_test = data.train_test_split(
        **cfg["train_test"])

    # environment
    logging.info("Environment creation...")
    train_env = AnomalyDetectionEnv(X_train, y_train, **cfg["env"])
    test_env = AnomalyDetectionEnv(X_test, y_test, **cfg["env"])

    # agent & dqn stuff
    logging.info("Creating policy and target network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 2 if cfg["data"]["label_type"] == "last" else X_train.shape[1]
    model_kwargs = {"n_features": X_train.shape[2],
                    "output_dim": output_dim}
    model_kwargs.update(cfg["model"])

    # softmax + BCELoss (since our network does not have softmax layer at the end)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    solver = DQNSolver(env=train_env, model=DQN,
                       loss_fn=loss_fn, model_kwargs=model_kwargs, optimizer=Adam, optimizer_kwargs=cfg["optimizer"], **cfg["solver"])
    solver.fit()


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
