from pathlib import Path

import matplotlib.pyplot
import numpy as np


def plot_csv(
    csv_paths: str | list[str],
    plot_name: str | None = None,
    show: bool = True,
    output_path: str | None = None,
) -> None:
    
    csv_paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths
    if len(csv_paths) == 0:
        raise ValueError("csv_paths is empty")

    figure, axes = matplotlib.pyplot.subplots(
        nrows=len(csv_paths),
        ncols=1,
        figsize=(10, max(5, 3 * len(csv_paths))),
        squeeze=False,
    )

    axes = axes.ravel()

    for axis, csv_path in zip(axes, csv_paths):
        steps, values = read_csv(csv_path)
        title = Path(csv_path).stem
        draw_data(axis, steps, values, title)

    figure.tight_layout()
    figure.subplots_adjust(hspace=0.45)
    if plot_name is not None:
        figure.canvas.manager.set_window_title(plot_name)

    # save plot
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)

    # show plot
    if show:
        matplotlib.pyplot.show()

    matplotlib.pyplot.close(figure)


def read_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"csv not found: {csv_file}")

    data = np.genfromtxt(
        csv_file,
        delimiter=",",
        names=True,
        dtype=np.float64,
        encoding="utf-8-sig",
    )

    if data.size == 0:
        raise ValueError(f"csv has no rows: {csv_file}")

    names = data.dtype.names
    if names is None or len(names) < 2:
        raise ValueError(f"csv must have step column and value columns: {csv_file}")

    steps = np.asarray(data[names[0]], dtype=np.float64)

    values = np.vstack([
        np.asarray(data[name], dtype=np.float64)
        for name in names[1:]
    ]).T

    return steps, values


def draw_data(
    axis,
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
):
    low, mid, high = compute_interval(values)

    axis.fill_between(
        steps,
        low,
        high,
        alpha=0.20,
        color="tab:blue",
    )

    axis.plot(
        steps,
        mid,
        linewidth=1.5,
        color="tab:blue",
    )

    axis.set_title(title)
    axis.grid(alpha=0.3)


def compute_interval(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.nanpercentile(values, 25.0, axis=1)
    mid = np.nanpercentile(values, 50.0, axis=1)
    high = np.nanpercentile(values, 75.0, axis=1)
    return low, mid, high
