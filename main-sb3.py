# %%
import argparse
import os
from os.path import join
import json
import logging
from time import time
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler

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
from utils import make_metrics


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

    log_folder = join("logs", cfg["name"])
    os.makedirs(log_folder, exist_ok=True)

    logger_name = join(log_folder, f"{cfg['name']}.log")
    logging.basicConfig(
        handlers=[logging.FileHandler(
            filename=logger_name, mode="w"), logging.StreamHandler()],
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO)
    with open(join(log_folder, "config.json"), "w") as f:
        json.dump(cfg, f)
    logging.info(
        f"Configuration file copied in {join(log_folder, 'config.json')}:\n" + json.dumps(cfg, indent=4))

    # load and prepare dataset
    logging.info("Prepraing the dataset...")
    data = TSDataset(join("data", "rastro.tar.gz"), **cfg["data"])
    scaler = MinMaxScaler() if cfg["train_test"].pop("scaler") else None
    X_train, X_test, y_train, y_test = data.train_test_split(
        scaler=scaler, **cfg["train_test"])

    # environment
    logging.info("Environment creation...")
    train_env = Monitor(AnomalyDetectionEnv(
        X_train, y_train, **cfg["env"]), filename=join(log_folder, "train_monitor.csv"))
    test_env = Monitor(AnomalyDetectionEnv(
        X_test, y_test, **cfg["env"]), filename=join(log_folder, "test_monitor.csv"))

    logging.info("Validating environment with stable_baselines...")
    check_env(train_env)
    check_env(test_env)

    # agent & dqn stuff
    logging.info("Creating DQN agent...")
    features_extractor_kwargs = cfg["feature_extractor"]
    policy_kwargs = {
        "features_extractor_class": FeatureExtractor,
        "features_extractor_kwargs": features_extractor_kwargs
    }
    new_logger = configure(log_folder, ["csv", "tensorboard"])
    model = DQN("MlpPolicy", env=train_env,
                policy_kwargs=policy_kwargs, **cfg["policy"], verbose=2)
    model.set_logger(new_logger)

    reward_train, _ = evaluate_policy(
        model, train_env, n_eval_episodes=1, deterministic=True)
    reward_test, _ = evaluate_policy(
        model, test_env, n_eval_episodes=1, deterministic=True)
    logging.info(
        f"Reward before training:\n" +
        f"Training env = {reward_train:.2f}\n" +
        f"Test env     = {reward_test:.2f}"
    )

    train_metrics = make_metrics(model, train_env)
    test_metrics = make_metrics(model, test_env)

    logging.info(
        "Before Training\n" +
        "---------------\n" +
        "Training Env\n" +
        json.dumps(train_metrics, indent=4) + "\n"
        "Test Env\n" +
        json.dumps(test_metrics, indent=4)
    )

    logging.info("Training the model...")

    t1 = time()
    model.learn(**cfg["learn"])
    t2 = time()

    logging.info(f"Took {timedelta(seconds=t2-t1)}")

    reward_train, _ = evaluate_policy(model, train_env, n_eval_episodes=1)
    reward_test, _ = evaluate_policy(model, test_env, n_eval_episodes=1)
    logging.info(
        f"Reward after training:\n" +
        f"Training env = {reward_train:.2f}\n" +
        f"Test env     = {reward_test:.2f}"
    )

    train_metrics = make_metrics(model, train_env)
    test_metrics = make_metrics(model, test_env)

    logging.info(
        "After Training\n" +
        "--------------\n" +
        "Training Env\n" +
        json.dumps(train_metrics, indent=4) + "\n"
        "Test Env\n" +
        json.dumps(test_metrics, indent=4)
    )

    # plot_results(["logs"], cfg["learn"]["total_timesteps"],
    #             results_plotter.X_TIMESTEPS, "Anomaly Detection")


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
