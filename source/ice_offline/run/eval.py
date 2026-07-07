import csv
from collections.abc import Callable
from pathlib import Path

from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.dataset._types import Episode


EvalBatches = list[tuple[int, list[Episode]]]
EvalRows = list[tuple[int, list[float]]]
EvalTable = tuple[str, EvalRows]
EvalData = dict[str, EvalTable]


def write_eval(
    mode: str,
    task_id: str,
    batches: EvalBatches,
) -> tuple[Path, Path]:
    return write_eval_data(mode, task_id, eval_data(batches))


def eval_returns(batches: EvalBatches) -> EvalRows:
    return _rows(batches, _episode_return)


def eval_steps(batches: EvalBatches) -> EvalRows:
    return _rows(batches, _episode_length)


def eval_data(batches: EvalBatches) -> EvalData:
    return {
        "returns": ("step", eval_returns(batches)),
        "steps": ("step", eval_steps(batches)),
    }


def write_eval_data(
    mode: str,
    task_id: str,
    data: EvalData,
) -> tuple[Path, Path]:
    returns_output_path = returns_path(mode, task_id)
    steps_output_path = steps_path(mode, task_id)
    _write_csv(returns_output_path, data["returns"])
    _write_csv(steps_output_path, data["steps"])
    return returns_output_path, steps_output_path


def write_eval_rows(
    mode: str,
    task_id: str,
    returns_rows: EvalRows,
    steps_rows: EvalRows,
) -> tuple[Path, Path]:
    return write_eval_data(
        mode,
        task_id,
        {
            "returns": ("step", returns_rows),
            "steps": ("step", steps_rows),
        },
    )


def read_eval(mode: str, task_id: str) -> EvalData:
    return {
        "returns": _read_csv(returns_path(mode, task_id)),
        "steps": _read_csv(steps_path(mode, task_id)),
    }


def _read_csv(path: Path) -> EvalTable:
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


def _write_csv(output_path: Path, eval_table: EvalTable) -> Path:
    key, eval_rows = eval_table
    columns = max(len(values) for _, values in eval_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([key] + [str(index) for index in range(1, columns + 1)])
        for step, values in eval_rows:
            padding = ["nan"] * (columns - len(values))
            writer.writerow([step] + values + padding)
    return output_path


def _episode_return(episode: Episode) -> float:
    return float(episode.rewards.sum())


def _episode_length(episode: Episode) -> float:
    return float(len(episode.rewards))
