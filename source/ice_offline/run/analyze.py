import csv
from collections.abc import Callable
from pathlib import Path

from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.dataset._types import Episode


EvalBatches = list[tuple[int, list[Episode]]]
EvalRows = list[tuple[int, list[float]]]
EvalTable = tuple[str, EvalRows]

def analyze_returns(task_id: str, batches: EvalBatches) -> Path:
    rows = _rows(batches, _episode_return)
    return write_csv(returns_path(task_id), "step", rows)


def analyze_steps(task_id: str, batches: EvalBatches) -> Path:
    rows = _rows(batches, _episode_length)
    return write_csv(steps_path(task_id), "step", rows)


def read_csv(path: Path) -> EvalTable:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [
            (
                int(row[0]),
                [
                    float(value)
                    for value in row[1:]
                    if value != "" and value != "nan"
                ],
            )
            for row in reader
        ]
    return header[0], rows


def _rows(
    batches: EvalBatches,
    value_fn: Callable[[Episode], float],
) -> EvalRows:
    return [
        (step, [value_fn(episode) for episode in episodes])
        for step, episodes in batches
    ]


def write_csv(output_path: Path, key: str, rows: EvalRows) -> Path:
    columns = max(len(values) for _, values in rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([key] + [str(index) for index in range(1, columns + 1)])
        for step, values in rows:
            padding = ["nan"] * (columns - len(values))
            writer.writerow([step] + values + padding)
    return output_path


def _episode_return(episode: Episode) -> float:
    return float(episode.rewards.sum())


def _episode_length(episode: Episode) -> float:
    return float(len(episode.rewards))
