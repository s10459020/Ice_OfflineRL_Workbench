import csv
import json
from pathlib import Path


def table_true(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_paths: list[list[Path]],
    lower_paths: list[Path | float | None],
    upper_paths: list[Path | float | None],
    output_path: Path,
) -> Path:
    rows = []
    for index, dataset_id in enumerate(dataset_ids):
        values = [_mean(path) for path in data_paths[index]]
        lower = _bound(lower_paths[index], values, min, _mean)
        upper = _bound(upper_paths[index], values, max, _mean)
        rows.append([dataset_id, _cell(lower), *[_cell(value) for value in values], _cell(upper)])
    return _write(output_path, ["task", "lower", *agent_ids, "upper"], rows)


def table_mean(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_paths: list[list[Path]],
    lower_paths: list[Path | float | None],
    upper_paths: list[Path | float | None],
    output_path: Path,
) -> Path:
    rows = []
    for index, dataset_id in enumerate(dataset_ids):
        values = [_mean(path) for path in data_paths[index]]
        lower = _bound(lower_paths[index], values, min, _mean)
        upper = _bound(upper_paths[index], values, max, _mean)
        rows.append([dataset_id, *[_cell(_scale(value, lower, upper)) for value in values]])
    return _write(output_path, ["task", *agent_ids], rows)


def table_pr95(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_paths: list[list[Path]],
    lower_paths: list[Path | float | None],
    upper_paths: list[Path | float | None],
    output_path: Path,
) -> Path:
    rows = []
    for index, dataset_id in enumerate(dataset_ids):
        values = [_pr95(path) for path in data_paths[index]]
        lower = _bound(lower_paths[index], values, min, _pr95)
        upper = _bound(upper_paths[index], values, max, _pr95)
        rows.append([dataset_id, *[_cell(_scale(value, lower, upper)) for value in values]])
    return _write(output_path, ["task", *agent_ids], rows)


def _mean(path: Path) -> float:
    values = _read(path)
    return sum(values) / len(values)


def _pr95(path: Path) -> float:
    values = _read(path)
    values.sort()
    index = (len(values) - 1) * 0.95
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _read(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as file:
        return [float(value) for value in json.load(file)]


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved: {path}")
    return path


def _cell(value: float) -> str:
    return str(float(value))


def _scale(value: float, lower: float, upper: float) -> float:
    return (value - lower) / (upper - lower) * 100.0


def _bound(value: Path | float | None, values: list[float], bound_fn, reduce_fn) -> float:
    if value is None:
        return bound_fn(values)
    if isinstance(value, Path):
        return reduce_fn(value)
    return float(value)
