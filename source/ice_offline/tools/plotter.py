import csv
from pathlib import Path

import matplotlib.pyplot
import numpy as np

def plot(
    metrics_path: str,
    eval_paths: str,
    output_path: str,
) -> None:
    metrics_steps, metrics_values, metric_names = read_metric_csv(metrics_path)
    eval_steps, eval_values = read_eval_csv(eval_paths) 

    metric_rows = (len(metric_names) + 1) // 2
    rows = metric_rows + 1
    figure = matplotlib.pyplot.figure(figsize=(14, rows * 4))
    grid = figure.add_gridspec(rows, 2)

    for i, metric_name in enumerate(metric_names):
        axis = figure.add_subplot(grid[i//2, i%2])
        draw_metric(axis, metric_name, metrics_steps, metrics_values[:, i])

    axis = figure.add_subplot(grid[metric_rows, :])
    draw_eval(axis, "returns", eval_steps, eval_values)

    figure.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file)
    matplotlib.pyplot.close(figure)

def read_metric_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    names = rows[0][1:]

    data = np.array(rows[1:], dtype=np.float64)
    steps = data[:, 0]
    values = data[:, 1:]
    return steps, values, names

def read_eval_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    data = np.array(rows[1:], dtype=np.float64)
    steps = data[:, 0]
    values = data[:, 1:]
    return steps, values

def draw_eval(
    axis,
    title: str,
    steps: np.ndarray,
    values: np.ndarray,
):
    p25 = np.nanpercentile(values, 25.0, axis=1)
    p50 = np.nanpercentile(values, 50.0, axis=1)
    p75 = np.nanpercentile(values, 75.0, axis=1)

    axis.fill_between(steps, p25, p75, alpha=0.20, color="tab:red")
    axis.plot(steps, p50, linewidth=1.5, color="tab:red")
    axis.set_title(title)
    axis.grid(alpha=0.3)

def draw_metric(
    axis,
    title: str,
    steps: np.ndarray,
    values: np.ndarray,
):
    axis.plot(steps, values, linewidth=1.5, color="tab:blue")
    axis.set_title(title)
    axis.grid(alpha=0.3)
