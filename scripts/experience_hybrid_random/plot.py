from pathlib import Path

import numpy as np
import ice_offline.run.plot

from ice_offline.config.paths import _task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.eval import EvalRows
from ice_offline.run.eval import cal_eval
from ice_offline.run.eval import eval_returns
from ice_offline.run.eval import eval_steps
from ice_offline.run.eval import write_eval_rows

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl",
    "sdc",
    "sdc_cql",
    "scas",
    "scaspl",
]

MODELS = [
    "scas_model",
    "sdc_model",
]


def plot(task_id: str, rows: EvalRows) -> Path:
    dataset_id, agent_id, _ = task_id.rsplit("-", 2)
    series_list = [
        (
            str(step),
            np.arange(1, len(values) + 1, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        for step, values in rows
    ]
    output_path = plot_path("dataset", dataset_id, agent_id)
    path = ice_offline.run.plot.plot_overlay(task_id, series_list, output_path)
    print(f"saved: {path}")
    return path


def eval(task_id: str, eval_path: Path) -> EvalRows:
    eval_dataset = EvalDataset(path=eval_path, device="cpu")
    batches = eval_dataset.batch_episodes
    returns_rows = eval_returns(batches)
    steps_rows = eval_steps(batches)
    returns_output_path, steps_output_path = write_eval_rows("test", task_id, returns_rows, steps_rows)
    print(f"saved: {returns_output_path}")
    print(f"saved: {steps_output_path}")
    return returns_rows


def plot_item(dataset_id: str, id: str, eval_output_paths: list[str]) -> None:
    task_id = _task_id(dataset_id, id)
    metrics_output_path = metric_path(task_id)
    output_path = plot_path("train", dataset_id, id)

    print(f"plot dataset={dataset_id}, id={id}")
    ice_offline.run.plot.plot([metrics_output_path], eval_output_paths, output_path)
    print(f"saved: {output_path}")


def plot_agent(dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    returns_output_path, steps_output_path = cal_eval(task_id, "train")
    plot_item(dataset_id, agent_id, [returns_output_path, steps_output_path])


def plot_model(dataset_id: str, model_id: str) -> None:
    plot_item(dataset_id, model_id, [])


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for agent_id in AGENTS:
            plot_agent(dataset_id, agent_id)
        for model_id in MODELS:
            plot_model(dataset_id, model_id)
