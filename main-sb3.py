# %%
import argparse
import os
from os.path import join, splitext, basename
import json
import logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

import torch
from torch.optim import Adam

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

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
        handlers=[logging.FileHandler(
            filename=f"{logger_name}.log", mode="w"), logging.StreamHandler()],
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO)
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
    train_env = Monitor(AnomalyDetectionEnv(
        X_train, y_train, **cfg["env"]), filename="sb_logs")
    test_env = Monitor(AnomalyDetectionEnv(
        X_test, y_test, **cfg["env"]), filename="sb_logs")

    logging.info("Validating environment with stable_baselines...")
    check_env(train_env)

    # agent & dqn stuff
    logging.info("Creating policy and target network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output_dim = 2 if cfg["data"]["label_type"] == "last" else X_train.shape[1]
    features_extractor_kwargs = cfg["feature_extractor"]
    policy_kwargs = {
        "features_extractor_class": FeatureExtractor,
        "features_extractor_kwargs": features_extractor_kwargs
    }
    new_logger = configure("sb_logs", ["csv", "tensorboard"])
    model = DQN("MlpPolicy", env=train_env,
                policy_kwargs=policy_kwargs, **cfg["policy"], verbose=2)
    model.set_logger(new_logger)

    reward_train, _ = evaluate_policy(model, train_env, n_eval_episodes=1)
    reward_test, _ = evaluate_policy(model, test_env, n_eval_episodes=1)
    logging.info(
        f"Reward before training:\n" +
        f"Training env = {reward_train:.2f}\n" +
        f"Test env     = {reward_test:.2f}"
    )

    model.learn(**cfg["learn"])

    reward_train, _ = evaluate_policy(model, train_env, n_eval_episodes=1)
    reward_test, _ = evaluate_policy(model, test_env, n_eval_episodes=1)
    logging.info(
        f"Reward after training:\n" +
        f"Training env = {reward_train:.2f}\n" +
        f"Test env     = {reward_test:.2f}"
    )

    logging.info("Testing the model...")

    y_pred = []
    state = test_env.reset()
    while True:
        y, _ = model.predict(state)
        y_pred.append(y)
        state, _, done, _ = test_env.step(y)
        if done:
            break

    logging.info(f"Accuracy = {balanced_accuracy_score(y_test, y_pred)}")
    logging.info(f"Cm       = {confusion_matrix(y_test, y_pred)}")
    logging.info(f"F1       = {f1_score(y_test, y_pred)}")

    # plot_results(["logs"], cfg["learn"]["total_timesteps"],
    #             results_plotter.X_TIMESTEPS, "Anomaly Detection")


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
