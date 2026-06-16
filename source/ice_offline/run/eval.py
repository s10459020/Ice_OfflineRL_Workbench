import csv
import json
from pathlib import Path
from typing import Callable

from ice_offline.dataset._types import Episode
from ice_offline.store.eval.loader import EvalLoader


# ====================
# Public API
# ====================
def cal_returns(input_path: Path, output_path: Path) -> Path:
    return _cal(input_path, output_path, _return_value)


def cal_steps(input_path: Path, output_path: Path) -> Path:
    return _cal(input_path, output_path, _step_value)


def cal_final_returns(input_path: Path, output_path: Path) -> Path:
    return _cal_final(input_path, output_path, _return_value)


def cal_final_steps(input_path: Path, output_path: Path) -> Path:
    return _cal_final(input_path, output_path, _step_value)


# ====================
# Private Methods
# ====================
def _cal(
    input_path: Path,
    output_path: Path,
    value_fn: Callable[[Episode], float],
) -> Path:
    loader = EvalLoader(input_path)
    rows = _rows(loader, value_fn)
    _write_csv(output_path, rows)
    return output_path


def _rows(
    loader: EvalLoader,
    value_fn: Callable[[Episode], float],
) -> list[tuple[int, list[float]]]:
    return [
        (step, [value_fn(episode) for episode in episodes])
        for step, episodes in loader.load_batch_episodes()
    ]


def _write_csv(output_path: Path, rows: list[tuple[int, list[float]]]) -> None:
    columns = max(len(values) for _, values in rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + [str(index) for index in range(columns)])
        for step, values in rows:
            padding = ["nan"] * (columns - len(values))
            writer.writerow([step] + values + padding)


def _cal_final(
    input_path: Path,
    output_path: Path,
    value_fn: Callable[[Episode], float],
) -> Path:
    loader = EvalLoader(input_path)
    step, episodes = loader.load_batch_episodes()[-1]
    _ = step
    values = [value_fn(episode) for episode in episodes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(values, file)
    return output_path


def _return_value(episode: Episode) -> float:
    return float(episode.rewards.sum())


def _step_value(episode: Episode) -> float:
    return float(len(episode.rewards))
