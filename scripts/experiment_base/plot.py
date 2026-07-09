from pathlib import Path

import numpy as np

from ice_offline.config.paths import eval_path
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import model_path
from ice_offline.config.paths import plot_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.config.paths import task_id
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.analyze import analyze_returns
from ice_offline.run.analyze import analyze_steps
from ice_offline.run.analyze import read_csv
from ice_offline.run.plot import plot_multi
from ice_offline.run.plot import plot_overlay

EXPERIMENT = "base"
EXPERIMENT_TRAIN = "base_train"

DATASETS = [
    "hopper_d4rl_medium",
    "hopper_d4rl_hybrid",
    "hopper_d4rl_expert",
    "hopper_replay_medium",
    "hopper_replay_expert",
    "halfcheetah_d4rl_medium",
    "halfcheetah_d4rl_hybrid",
    "halfcheetah_d4rl_expert",
    "halfcheetah_replay_medium",
    "halfcheetah_replay_expert",
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
]

AGENTS = [
    ("bc", 50_000),
    ("td3bc_n", 100_000),
    ("iql", 200_000),
    ("cql", 500_000),
    ("aspl_gp", 500_000),
    ("scas_gp", 500_000),
    ("scaspl_gp", 500_000),
]

MODELS = [
    "scas_model",
]

def analyze(task_id: str, eval_path: Path) -> None:
    eval_dataset = EvalDataset(path=eval_path, device="cpu")
    batches = eval_dataset.batch_episodes
    path = analyze_returns(task_id, batches)
    print(f"saved: {path}")
    path = analyze_steps(task_id, batches)
    print(f"saved: {path}")

def plot_test(task_id: str, returns_path: Path) -> Path:
    rows = read_csv(returns_path)[1]
    series_list = [
        (
            str(step),
            np.arange(1, len(values) + 1, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        for step, values in rows
    ]
    path = plot_path(task_id)
    plot_overlay(task_id, series_list, path)
    print(f"saved: {path}")
    return path

def plot_train(task_id: str, metrics_path: Path, analyze_paths: list[Path]) -> Path:
    path = plot_path(task_id)
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
        for agent_id, step in AGENTS:
            id = task_id(dataset_id, agent_id, EXPERIMENT_TRAIN)
            train_eval_path = eval_path(id)
            train_metric_path = metric_path(id)
            if _skip(model_path(id, step), train_eval_path, train_metric_path):
                continue

            analyze(id, train_eval_path)
            plot_train(id, train_metric_path, [returns_path(id), steps_path(id)])
            
        for model_id in MODELS:
            id = task_id(dataset_id, model_id, EXPERIMENT_TRAIN)
            train_metric_path = metric_path(id)
            if _skip(model_path(id, 100_000), train_metric_path):
                continue

            plot_train(id, train_metric_path, [])

    for dataset_id in DATASETS:
        for agent_id, step in AGENTS:
            train_id = task_id(dataset_id, agent_id, EXPERIMENT_TRAIN)
            test_id = task_id(dataset_id, agent_id, EXPERIMENT)
            test_eval_path = eval_path(test_id)
            if _skip(model_path(train_id, step), test_eval_path):
                continue
            
            analyze(test_id, test_eval_path)
            plot_test(test_id, returns_path(test_id))
