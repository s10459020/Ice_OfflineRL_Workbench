import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def boxplot(
    title: str,
    members: list[tuple[str, Path | None]],
    output_path: Path,
) -> Path | None:
    labels: list[str] = []
    values: list[list[float]] = []

    for label, path in members:
        if path is None or not path.exists():
            continue
        member_values = _read_values(path)
        if not member_values:
            continue
        labels.append(label)
        values.append(member_values)

    if not values:
        return None

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.boxplot(values, tick_labels=labels, showfliers=True)
    axis.set_title(title)
    axis.set_ylabel("Return")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _read_values(path: Path) -> list[float]:
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
