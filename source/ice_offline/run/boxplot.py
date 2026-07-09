import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import steps_path


def boxplot(
    title: str,
    members: list[tuple[str, Path | None]],
    output_path: Path,
) -> None:
    labels: list[str] = []
    values: list[list[float]] = []
    steps: list[list[float]] = []

    for label, path in members:
        if path is None or not path.exists():
            continue
        member_values, member_steps = _read_member(path)
        if not member_values:
            continue
        labels.append(label)
        values.append(member_values)
        steps.append(member_steps)

    _save_boxplot(title, labels, values, steps, output_path)


def boxplot_data(
    title: str,
    labels: list[str],
    values: list[list[float] | None],
    output_path: Path,
) -> None:
    filtered_labels: list[str] = []
    filtered_values: list[list[float]] = []
    steps: list[list[float]] = []

    for label, member_values in zip(labels, values):
        if not member_values:
            continue
        filtered_labels.append(label)
        filtered_values.append(member_values)
        steps.append([1.0] * len(member_values))

    _save_boxplot(title, filtered_labels, filtered_values, steps, output_path)


def write_boxplots(
    group: str,
    dataset_ids: list[str],
    agent_ids: list[str],
    data_values: list[list[list[float] | None]],
    lower_values: list[list[float] | None],
    upper_values: list[list[float] | None],
) -> None:
    for index, dataset_id in enumerate(dataset_ids):
        labels = ["lower", *agent_ids, "upper"]
        values = [lower_values[index], *data_values[index], upper_values[index]]
        output_path = VIEW_ROOT / "boxplot" / group / f"{dataset_id}.png"
        boxplot_data(dataset_id, labels, values, output_path)


def _save_boxplot(
    title: str,
    labels: list[str],
    values: list[list[float]],
    steps: list[list[float]],
    output_path: Path,
) -> None:
    if not values:
        return

    figure, axis = plt.subplots(figsize=(14, 6))
    for index, (member_values, member_steps) in enumerate(zip(values, steps), start=1):
        _draw_step_weighted_violin(axis, index, member_values, member_steps)

    axis.boxplot(
        values,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        widths=0.18,
        boxprops={"facecolor": "#4C72B0", "alpha": 0.5},
        whiskerprops={"color": "#1F3A5F"},
        capprops={"color": "#1F3A5F"},
        medianprops={"color": "#1F3A5F", "linewidth": 1.5},
    )
    axis.set_title(title)
    axis.set_ylabel("Return")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"saved: {output_path}")


def _read_member(path: Path) -> tuple[list[float], list[float]]:
    mode = path.parent.name
    task_id = path.stem
    step_csv_path = steps_path(mode, task_id)
    if not step_csv_path.exists():
        values = _read_csv_values(path)
        return values, [1.0] * len(values)

    returns = _read_csv_values(path)
    step_values = _read_csv_values(step_csv_path)
    count = min(len(returns), len(step_values))
    return returns[:count], step_values[:count]


def _read_csv_values(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        values: list[float] = []
        for row in reader:
            for value in row[1:]:
                if value == "" or value == "nan":
                    continue
                values.append(float(value))
    return values


def _draw_step_weighted_violin(
    axis,
    position: int,
    member_values: list[float],
    member_steps: list[float],
) -> None:
    values = np.asarray(member_values, dtype=np.float64)
    steps = np.asarray(member_steps, dtype=np.float64)
    value_min = float(values.min())
    value_max = float(values.max())
    value_span = value_max - value_min
    padding = max(value_span * 0.02, 1.0)
    window_low = max(value_span * 0.03, 1.0)
    window_high = max(value_span * 0.03, 1.0)
    ys = np.linspace(value_min - padding, value_max + padding, 2048)
    profile = np.asarray(
        [
            steps[((values >= y - window_low) & (values <= y + window_high))].sum()
            for y in ys
        ],
        dtype=np.float64,
    )
    max_window_steps = float(profile.max())
    if max_window_steps <= 0:
        return

    half_width = 0.4 * profile / max_window_steps
    axis.fill_betweenx(
        ys,
        position - half_width,
        position + half_width,
        facecolor="#4C72B0",
        edgecolor="#4C72B0",
        alpha=0.2,
        linewidth=0.8,
    )
