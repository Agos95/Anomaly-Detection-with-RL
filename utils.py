# %%
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from env import AnomalyDetectionEnv
from stable_baselines3.common.base_class import BaseAlgorithm
# %%


def make_metrics(model: BaseAlgorithm, env: AnomalyDetectionEnv):
    """
    Calculate metrics.

    Parameters
    ----------
    model : BaseAlgorithm
        Stable Baseline 3 RL algorithm.
    env : AnomalyDetectionEnv
        Custom Gym env for anomaly detection.

    Returns
    -------
    metrics : dict
        Dictionary with: accuracy, balanced accuracy, f1,
        confusion matrix (also normalized for true label).
    """
    y_true, y_pred = [], []

    state = env.reset()
    while True:
        yp, _ = model.predict(state, deterministic=True)
        y_pred.append(yp)
        state, _, done, info = env.step(yp)
        yt = info["true_action"][0]
        y_true.append(yt)
        if done:
            break

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
        "cm_norm": confusion_matrix(y_true, y_pred, normalize="true").tolist()
    }

    return metrics
