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

EXPERIMENT = "experience_hybrid_random"
EXPERIMENT_TRAIN = "experience_hybrid_random_train"

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("aspl_c", None, 500_000),
    ("scas_gp", 500_000, 500_000),
    ("scaspl_n", 500_000, 500_000),
    ("scc_n", 500_000, 500_000),
]

MODELS = [
    ("scas_model", 500_000),
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

        for model_id, model_step in MODELS:
            id = experiment_task_id(EXPERIMENT_TRAIN, model_id, dataset_id)
            train_metric_path = metric_path(id)
            if _skip(model_path(id, model_step), train_metric_path):
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
