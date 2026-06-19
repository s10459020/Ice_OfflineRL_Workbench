import csv
from pathlib import Path
from typing import Callable

from ice_offline.config.paths import eval_data_path
from ice_offline.config.paths import main_data_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.dataset._types import Episode
from ice_offline.store.eval.loader import EvalLoader
from ice_offline.store.minari.loader import MinariLoader


# ====================
# Public API
# ====================
def cal_eval(task_id: str, mode: str = "train") -> tuple[Path, Path]:
    input_path = eval_data_path(mode, task_id)
    returns_output_path = returns_path(mode, task_id)
    steps_output_path = steps_path(mode, task_id)
    loader = EvalLoader(input_path)
    batches = loader.load_batch_episodes()
    _write_csv(returns_output_path, "episode", _rows(batches, _return_value))
    _write_csv(steps_output_path, "episode", _rows(batches, _step_value))
    return returns_output_path, steps_output_path


def cal_main(task_id: str, mode: str = "test") -> tuple[Path, Path]:
    input_path = main_data_path(mode, task_id)
    returns_output_path = returns_path(mode, task_id)
    steps_output_path = steps_path(mode, task_id)
    loader = MinariLoader(input_path, device="cpu")
    batches = [(1, loader.load_episodes())]
    _write_csv(returns_output_path, "episode", _rows(batches, _return_value))
    _write_csv(steps_output_path, "episode", _rows(batches, _step_value))
    return returns_output_path, steps_output_path


def cal_dataset(dataset_id: str, mode: str = "dataset", device: str = "cuda") -> tuple[Path, Path]:
    dataset = make_dataset(dataset_id, device=device)
    returns_output_path = returns_path(mode, dataset_id)
    steps_output_path = steps_path(mode, dataset_id)
    batches = [(1, dataset.episodes)]
    _write_csv(returns_output_path, "episode", _rows(batches, _return_value))
    _write_csv(steps_output_path, "episode", _rows(batches, _step_value))
    return returns_output_path, steps_output_path


def _rows(
    batches: list[tuple[int, list[Episode]]],
    value_fn: Callable[[Episode], float],
) -> list[tuple[int, list[float]]]:
    return [
        (step, [value_fn(episode) for episode in episodes])
        for step, episodes in batches
    ]


def _write_csv(output_path: Path, key: str, rows: list[tuple[int, list[float]]]) -> None:
    columns = max(len(values) for _, values in rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([key] + [str(index) for index in range(1, columns + 1)])
        for step, values in rows:
            padding = ["nan"] * (columns - len(values))
            writer.writerow([step] + values + padding)


def _return_value(episode: Episode) -> float:
    return float(episode.rewards.sum())


def _step_value(episode: Episode) -> float:
    return float(len(episode.rewards))
