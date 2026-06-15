import csv
import json
from pathlib import Path

from ice_offline.config.paths import VIEW_ROOT


def table(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_paths: dict[str, dict[str, Path]],
    lower_paths: dict[str, Path],
    upper_paths: dict[str, Path],
    output_dir: Path | None = None,
) -> list[Path]:
    active_agent_ids = [agent_id for agent_id in agent_ids if agent_id != "random"]
    output_dir = output_dir or VIEW_ROOT / "returns"
    header = ["task", *active_agent_ids]

    return [
        _save_csv(
            output_dir / "true_returns.csv",
            ["task", "lower", *active_agent_ids, "upper"],
            _actual_rows(dataset_ids, active_agent_ids, data_paths, lower_paths, upper_paths),
        ),
        _save_csv(
            output_dir / "mean_returns.csv",
            header,
            _normalized_rows(dataset_ids, active_agent_ids, data_paths, lower_paths, upper_paths, _mean_return),
        ),
        _save_csv(
            output_dir / "pr95_returns.csv",
            header,
            _normalized_rows(dataset_ids, active_agent_ids, data_paths, lower_paths, upper_paths, _percentile_return),
        ),
    ]


def _read_values(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as file:
        return [float(value) for value in json.load(file)]


def _mean_return(path: Path) -> float:
    values = _read_values(path)
    return sum(values) / len(values)


def _percentile_return(path: Path, percentile: float = 95.0) -> float:
    values = _read_values(path)

    values.sort()
    index = (len(values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _actual_rows(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_paths: dict[str, dict[str, Path]],
    lower_paths: dict[str, Path],
    upper_paths: dict[str, Path],
) -> list[list[str]]:
    return [
        _actual_row(dataset_id, agent_ids, data_paths, lower_paths, upper_paths)
        for dataset_id in dataset_ids
    ]


def _normalized_rows(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_paths: dict[str, dict[str, Path]],
    lower_paths: dict[str, Path],
    upper_paths: dict[str, Path],
    return_fn,
) -> list[list[str]]:
    return [
        _normalized_row(dataset_id, agent_ids, data_paths, lower_paths, upper_paths, return_fn)
        for dataset_id in dataset_ids
    ]


def _actual_row(
    dataset_id: str,
    agent_ids: list[str],
    data_paths: dict[str, dict[str, Path]],
    lower_paths: dict[str, Path],
    upper_paths: dict[str, Path],
) -> list[str]:
    values = [_mean_return(lower_paths[dataset_id])]
    for agent_id in agent_ids:
        values.append(_mean_return(data_paths[dataset_id][agent_id]))
    values.append(_mean_return(upper_paths[dataset_id]))
    return [dataset_id, *[_cell(value) for value in values]]


def _normalized_row(
    dataset_id: str,
    agent_ids: list[str],
    data_paths: dict[str, dict[str, Path]],
    lower_paths: dict[str, Path],
    upper_paths: dict[str, Path],
    return_fn,
) -> list[str]:
    bottom = return_fn(lower_paths[dataset_id])
    top = _normalized_top(dataset_id, agent_ids, data_paths, upper_paths, return_fn)
    values = [
        _scale(return_fn(data_paths[dataset_id][agent_id]), bottom, top)
        for agent_id in agent_ids
    ]
    return [dataset_id, *[_cell(value) for value in values]]


def _normalized_top(
    dataset_id: str,
    agent_ids: list[str],
    data_paths: dict[str, dict[str, Path]],
    upper_paths: dict[str, Path],
    return_fn,
) -> float:
    return return_fn(upper_paths[dataset_id])


def _save_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved: {path}")
    return path


def _cell(value: float) -> str:
    return str(float(value))


def _scale(value: float, bottom: float, top: float) -> float:
    return (value - bottom) / (top - bottom) * 100.0
