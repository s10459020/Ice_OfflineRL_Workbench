from pathlib import Path

import numpy as np
import ice_offline.run.plot

from ice_offline.config.paths import plot_path
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.eval import EvalRows
from ice_offline.run.eval import eval_returns
from ice_offline.run.eval import eval_steps
from ice_offline.run.eval import write_eval_rows


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
