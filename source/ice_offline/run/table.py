import csv
from pathlib import Path

from ice_offline.config.paths import table_path


TableSeries = list[float]
TableCell = TableSeries | None
TableBound = TableSeries | float | None


def table_true(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_values: list[list[TableCell]],
    lower_values: list[TableBound],
    upper_values: list[TableBound],
    output_path: Path,
) -> Path:
    rows = []
    for index, dataset_id in enumerate(dataset_ids):
        values = [_mean(item) for item in data_values[index]]
        lower = _bound(lower_values[index], values, min, _mean)
        upper = _bound(upper_values[index], values, max, _mean)
        rows.append([dataset_id, _cell(lower), *[_cell(value) for value in values], _cell(upper)])
    return _write(output_path, ["task", "lower", *agent_ids, "upper"], rows)


def table_mean(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_values: list[list[TableCell]],
    lower_values: list[TableBound],
    upper_values: list[TableBound],
    output_path: Path,
) -> Path:
    rows = []
    for index, dataset_id in enumerate(dataset_ids):
        values = [_mean(item) for item in data_values[index]]
        lower = _bound(lower_values[index], values, min, _mean)
        upper = _bound(upper_values[index], values, max, _mean)
        rows.append([dataset_id, *[_cell(_scale(value, lower, upper)) if value is not None else "" for value in values]])
    return _write(output_path, ["task", *agent_ids], rows)


def table_pr95(
    dataset_ids: list[str],
    agent_ids: list[str],
    data_values: list[list[TableCell]],
    lower_values: list[TableBound],
    upper_values: list[TableBound],
    output_path: Path,
) -> Path:
    rows = []
    for index, dataset_id in enumerate(dataset_ids):
        values = [_pr95(item) for item in data_values[index]]
        lower = _bound(lower_values[index], values, min, _pr95)
        upper = _bound(upper_values[index], values, max, _pr95)
        rows.append([dataset_id, *[_cell(_scale(value, lower, upper)) if value is not None else "" for value in values]])
    return _write(output_path, ["task", *agent_ids], rows)


def write_tables(
    group: str,
    dataset_ids: list[str],
    agent_ids: list[str],
    data_values: list[list[TableCell]],
    lower_values: list[TableBound],
    upper_values: list[TableBound],
) -> tuple[Path, Path, Path]:
    return (
        table_true(
            dataset_ids,
            agent_ids,
            data_values,
            lower_values,
            upper_values,
            table_path(group, "true_returns.csv"),
        ),
        table_mean(
            dataset_ids,
            agent_ids,
            data_values,
            lower_values,
            upper_values,
            table_path(group, "mean_returns.csv"),
        ),
        table_pr95(
            dataset_ids,
            agent_ids,
            data_values,
            lower_values,
            upper_values,
            table_path(group, "pr95_returns.csv"),
        ),
    )


def _mean(values: TableCell) -> float | None:
    if values is None:
        return None
    return sum(values) / len(values)


def _pr95(values: TableCell) -> float | None:
    if values is None:
        return None
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * 0.95
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved: {path}")
    return path


def _cell(value: float | None) -> str:
    if value is None:
        return ""
    return str(float(value))


def _scale(value: float | None, lower: float, upper: float) -> float:
    if value is None:
        return 0.0
    return (value - lower) / (upper - lower) * 100.0


def _bound(value: TableBound, values: list[float | None], bound_fn, reduce_fn) -> float:
    valid_values = [item for item in values if item is not None]
    if value is None:
        return bound_fn(valid_values)
    if isinstance(value, list):
        return reduce_fn(value)
    return float(value)
