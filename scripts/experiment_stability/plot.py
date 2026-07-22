from pathlib import Path

import numpy as np

from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import model_path
from ice_offline.config.paths import plot_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.analyze import analyze_returns
from ice_offline.run.analyze import analyze_steps
from ice_offline.run.analyze import read_csv
from ice_offline.run.plot import plot_multi
from ice_offline.run.plot import plot_overlay

EXPERIMENT = "base"
EXPERIMENT_TRAIN = "base_train"

DATASETS = [
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
]

AGENTS = [
    # ("td3", None, 50_000),
    # ("td3_gamma_90", None, 50_000),
    # ("td3_n", None, 50_000),
    # ("td3_r", None, 50_000),
    # ("td3_gp", None, 50_000),
    # ("td3_gpn", None, 50_000),
    # ("td3bc", None, 100_000),
    # ("td3bc_n", None, 100_000),
    # ("td3bc_plus", None, 100_000),
    # ("td3bc_r", None, 100_000),
    # ("td3bc_gp", None, 100_000),
    # ("td3bc_gp_plus", None, 100_000),
    # ("td3bc_gpn", None, 100_000),
    # ("cql", None, 500_000),
    # ("cql_threshold_n5", None, 500_000),
    # ("cql_threshold_5", None, 500_000),
    # ("cql_gp", None, 500_000),
    # ("aspl", None, 200_000),
    # ("aspl_r", None, 200_000),
    # ("aspl_gamma_90", None, 200_000),
    # ("aspl_gamma_95", None, 200_000),
    # ("aspl_gp", None, 1_000_000),
    # ("scas", 100_000, 500_000),
    # ("scas_n", 100_000, 500_000),
    # ("scas_gp", 100_000, 500_000),
    # ("scas_gpn", 100_000, 500_000),
    # ("scaspl", 100_000, 500_000),
    # ("scaspl_n", 100_000, 500_000),
    # ("scaspl_gp", 100_000, 500_000),
    # ("scaspl_gpn", 100_000, 500_000),
]

MODELS = [
    # "scas_model",
]


def analyze(task_id: str, eval_path: Path) -> None:
    eval_dataset = EvalDataset(path=eval_path, device="cpu")
    batches = eval_dataset.batch_episodes
    path = analyze_returns(task_id, batches)
    print(f"saved: {path}")
    path = analyze_steps(task_id, batches)
    print(f"saved: {path}")


def plot_test(task_id: str, returns_path: Path, dataset_id: str, agent_id: str) -> Path:
    rows = read_csv(returns_path)[1]
    series_list = [
        (
            str(step),
            np.arange(1, len(values) + 1, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        for step, values in rows
    ]
    path = plot_path(EXPERIMENT, dataset_id, agent_id)
    plot_overlay(task_id, series_list, path)
    print(f"saved: {path}")
    return path


def plot_train(task_id: str, metrics_path: Path, analyze_paths: list[Path], dataset_id: str, agent_id: str) -> Path:
    path = plot_path(EXPERIMENT_TRAIN, dataset_id, agent_id)
    plot_multi(metrics_path, analyze_paths, path)
    print(f"saved: {path}")
    return path


def _skip(*paths: Path) -> bool:
    for path in paths:
        if path.exists():
            continue
        print(f"skip missing: {path}")
        return True
    return False


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for agent_id, _, agent_step in AGENTS:
            id = experiment_task_id(EXPERIMENT_TRAIN, agent_id, dataset_id)
            train_eval_path = eval_path(id)
            train_metric_path = metric_path(id)
            if _skip(model_path(id, agent_step), train_eval_path, train_metric_path):
                continue

            analyze(id, train_eval_path)
            plot_train(id, train_metric_path, [returns_path(id), steps_path(id)], dataset_id, agent_id)

        for model_id in MODELS:
            id = experiment_task_id(EXPERIMENT_TRAIN, model_id, dataset_id)
            train_metric_path = metric_path(id)
            if _skip(model_path(id, 100_000), train_metric_path):
                continue

            plot_train(id, train_metric_path, [], dataset_id, model_id)

    for dataset_id in DATASETS:
        for agent_id, _, agent_step in AGENTS:
            train_id = experiment_task_id(EXPERIMENT_TRAIN, agent_id, dataset_id)
            test_id = experiment_task_id(EXPERIMENT, agent_id, dataset_id)
            test_eval_path = eval_path(test_id)
            if _skip(model_path(train_id, agent_step), test_eval_path):
                continue

            analyze(test_id, test_eval_path)
            plot_test(test_id, returns_path(test_id), dataset_id, agent_id)
