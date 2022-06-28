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
from stable_baselines3 import DQN

from dataset import TSDataset
from env import AnomalyDetectionEnv
from stableBaselinev3.model import FeatureExtractor


# %%

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/config-sb3.json", help="Configuration file")
    args = parser.parse_args()
    return vars(args)

# %%


def main(args):

    with open(args["config"], "r") as f:
        cfg = json.load(f)

    logger_name = join("logs", basename(splitext(args["config"])[0]))
    logging.basicConfig(
        filename=f"{logger_name}.log", filemode="w", format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO)
    logging.info(
        f"Configuration file '{args['config']}':\n" + json.dumps(cfg, indent=4))

    # load and prepare dataset
    logging.info("Prepraing the dataset...")
    data = TSDataset(join("data", "rastro.tar.gz"), **cfg["data"])
    scaler = MinMaxScaler() if cfg["train_test"].pop("scaler") else None
    X_train, X_test, y_train, y_test = data.train_test_split(
        scaler=scaler, **cfg["train_test"])

    # environment
    logging.info("Environment creation...")
    train_env = AnomalyDetectionEnv(X_train, y_train, **cfg["env"])
    test_env = AnomalyDetectionEnv(X_test, y_test, **cfg["env"])

    # agent & dqn stuff
    logging.info("Creating policy and target network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 2 if cfg["data"]["label_type"] == "last" else X_train.shape[1]
    features_extractor_kwargs = cfg["feature_extractor"]
    features_extractor_kwargs["n_features"] = X_train.shape[-1]
    policy_kwargs = {
        "features_extractor_class": FeatureExtractor,
        "features_extractor_kwargs": features_extractor_kwargs
    }
    model = DQN("MlpPolicy", env=train_env,
                policy_kwargs=policy_kwargs, **cfg["policy"])


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
