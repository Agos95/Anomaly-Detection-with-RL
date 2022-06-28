# %%
import argparse
import os
from os.path import join, splitext, basename
import json
import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.optimizers import Adam
from keras.utils import plot_model
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from dataset import TSDataset
from env import AnomalyDetectionEnv
from kerasRL.model import kDQN, CustomProcessor

# %%


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/config-rl.json", help="Configuration file")
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
    output_dim = 2 if cfg["data"]["label_type"] == "last" else X_train.shape[1]
    model_kwargs = cfg["model"]
    model_kwargs["output_dim"] = output_dim

    model = kDQN(input_shape=X_train[0].shape, **model_kwargs)
    plot_model(model, show_shapes=True)

    optimizer = Adam(**cfg["optimizer"])
    memory = SequentialMemory(**cfg["memory"])
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), **cfg["policy"])

    agent = DQNAgent(model=model, nb_actions=train_env.action_space.n, processor=CustomProcessor(),
                     memory=memory, policy=policy, **cfg["agent"])
    agent.compile(optimizer, metrics=["accuracy"])
    agent.fit(train_env, nb_steps=50000, verbose=2)


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
