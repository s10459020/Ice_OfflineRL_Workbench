from pathlib import Path

import numpy as np

from ice_offline.config.paths import plot_path
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.analyze import analyze_returns
from ice_offline.run.analyze import analyze_steps
from ice_offline.run.analyze import read_csv
from ice_offline.run.plot import plot_overlay

EXPERIMENT = "noise_state"


def analyze(task_id: str, eval_path: Path) -> None:
    eval_dataset = EvalDataset(path=eval_path, device="cpu")
    batches = eval_dataset.batch_episodes
    path = analyze_returns(task_id, batches)
    print(f"saved: {path}")
    path = analyze_steps(task_id, batches)
    print(f"saved: {path}")


def plot(task_id: str, returns_path: Path, dataset_id: str, agent_id: str) -> Path:
    rows = read_csv(returns_path)[1]
    series_list = [
        (
            str(step),
            np.arange(1, len(values) + 1, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
        for step, values in rows
    ]
    output_path = plot_path(EXPERIMENT, dataset_id, agent_id)
    plot_overlay(task_id, series_list, output_path)
    print(f"saved: {output_path}")
    return output_path
