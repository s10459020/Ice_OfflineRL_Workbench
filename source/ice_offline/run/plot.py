import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot
import numpy as np

MetricSeries = tuple[str, np.ndarray, np.ndarray]
EvalSeries = tuple[str, np.ndarray, np.ndarray]
LineSeries = tuple[str, np.ndarray, np.ndarray]


# ====================
# Public API
# ====================
def plot_multi(
    metric_path: Path,
    eval_paths: list[Path],
    output_path: Path,
) -> Path:
    metrics = _read_metrics(metric_path)
    evals = _read_evals(eval_paths)
    return plot_data(metrics, evals, output_path)


def plot_data(
    metrics: list[MetricSeries],
    evals: list[EvalSeries],
    output_path: Path,
) -> Path:
    row_count = _row_count(len(metrics)) + _row_count(len(evals))

    figure = matplotlib.pyplot.figure(figsize=(14, row_count * 4))
    grid = figure.add_gridspec(row_count, 2)

    _draw_metrics(figure, grid, metrics)
    _draw_evals(figure, grid, _row_count(len(metrics)), evals)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    matplotlib.pyplot.close(figure)
    return output_path


def plot_overlay(
    title: str,
    series_list: list[LineSeries],
    output_path: Path,
) -> None:
    figure, axis = matplotlib.pyplot.subplots(figsize=(14, 6))
    colors = matplotlib.pyplot.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (label, steps, values) in enumerate(series_list):
        color = colors[index % len(colors)]
        axis.plot(steps, values, linewidth=1.8, color=color, label=label)

    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    matplotlib.pyplot.close(figure)

# ====================
# Private Methods
# ====================
def _read_metrics(metric_path: Path) -> list[MetricSeries]:
    metrics: list[MetricSeries] = []
    steps, values, names = _read_metric_csv(metric_path)
    for index, name in enumerate(names):
        metrics.append((name, steps, values[:, index]))
    return metrics


def _read_evals(eval_paths: list[Path]) -> list[EvalSeries]:
    return [
        (_eval_name(path), *_read_eval_csv(path))
        for path in eval_paths
    ]


def _read_metric_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    names = rows[0][1:]
    data = np.asarray(
        [
            [value if value != "" else np.nan for value in row]
            for row in rows[1:]
        ],
        dtype=np.float64,
    )
    return data[:, 0], data[:, 1:], names


def _read_eval_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    data = np.asarray(rows[1:], dtype=np.float64)
    return data[:, 0], data[:, 1:]


def _draw_metrics(
    figure,
    grid,
    metrics: list[MetricSeries],
) -> None:
    for index, (name, steps, values) in enumerate(metrics):
        axis = figure.add_subplot(grid[index // 2, index % 2])
        _draw_metric(axis, name, steps, values)


def _draw_evals(
    figure,
    grid,
    row_offset: int,
    evals: list[EvalSeries],
) -> None:
    for index, (name, steps, values) in enumerate(evals):
        axis = figure.add_subplot(grid[row_offset + index // 2, index % 2])
        _draw_eval(axis, name, steps, values)


def _draw_metric(
    axis,
    title: str,
    steps: np.ndarray,
    values: np.ndarray,
) -> None:
    keep = np.isfinite(values)
    smooth = _smooth_by_step(steps[keep], values[keep])
    axis.plot(steps[keep], values[keep], linewidth=0.5, alpha=0.25, color="tab:blue")
    axis.plot(steps[keep], smooth, linewidth=1.5, color="tab:blue")
    axis.set_title(title)
    axis.grid(alpha=0.3)


def _draw_eval(
    axis,
    title: str,
    steps: np.ndarray,
    values: np.ndarray,
) -> None:
    p25 = np.nanpercentile(values, 25.0, axis=1)
    p50 = np.nanpercentile(values, 50.0, axis=1)
    p75 = np.nanpercentile(values, 75.0, axis=1)

    axis.fill_between(steps, p25, p75, alpha=0.20, color="tab:red")
    axis.plot(steps, p50, linewidth=1.5, color="tab:red")
    axis.set_title(title)
    axis.grid(alpha=0.3)


def _row_count(count: int) -> int:
    return (count + 1) // 2


def _eval_name(path: Path) -> str:
    return path.parent.parent.name


def _smooth_by_step(steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(steps) == 0:
        return values

    window = (steps[-1] - steps[0]) / 500.0
    smooth = np.empty_like(values)
    left = 0
    total = 0.0
    count = 0
    for right, step in enumerate(steps):
        value = values[right]
        if np.isfinite(value):
            total += value
            count += 1

        while steps[left] < step - window:
            left_value = values[left]
            if np.isfinite(left_value):
                total -= left_value
                count -= 1
            left += 1

        smooth[right] = total / count if count > 0 else np.nan
    return smooth
