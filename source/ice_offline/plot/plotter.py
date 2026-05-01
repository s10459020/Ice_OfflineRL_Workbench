import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot
import numpy as np

from ice_offline.tools.paths import eval_plot_root


def plot_eval_csv(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    show_mean_line: bool = False,
    quantile_interval: tuple[float, float] | None = None,
    quantile_lines: Sequence[float] | None = None,
    reward_reference_lines: dict[str, float] | None = None,
) -> Path:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"csv not found: {csv_file}")

    with csv_file.open("r", encoding="utf-8", newline="") as csv_stream:
        reader = csv.DictReader(csv_stream)
        if not reader.fieldnames:
            raise ValueError(f"csv has no header: {csv_file}")

        metric_names = [name for name in reader.fieldnames if name != "step"]
        if not metric_names:
            raise ValueError(f"csv has no metric columns: {csv_file}")

        steps: list[float] = []
        metric_series: dict[str, list[float]] = {name: [] for name in metric_names}
        for row in reader:
            if row.get("step") is None:
                continue
            steps.append(_parse_float(row["step"]))
            for metric_name in metric_names:
                metric_value = row.get(metric_name)
                metric_series[metric_name].append(_parse_float(metric_value))

    if not steps:
        raise ValueError(f"csv has no rows: {csv_file}")

    if output_path is None:
        output_file = eval_plot_root() / f"{csv_file.stem}.png"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    ordered_metrics = _ordered_base_metrics(metric_names)
    figure, axes = matplotlib.pyplot.subplots(
        nrows=len(ordered_metrics),
        ncols=1,
        figsize=(10, max(3, 2.4 * len(ordered_metrics))),
        sharex=True,
    )
    if len(ordered_metrics) == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, ordered_metrics):
        q25_key = f"{metric_name}_q25"
        q50_key = f"{metric_name}_q50"
        q75_key = f"{metric_name}_q75"
        if q50_key in metric_series:
            values = np.asarray(metric_series[q50_key], dtype=np.float64)
            line_label = f"{metric_name}_q50"
        else:
            values = np.asarray(metric_series[metric_name], dtype=np.float64)
            line_label = metric_name
        finite_values = values[np.isfinite(values)]

        axis.plot(steps, values, linewidth=1.5, label=line_label)

        if q25_key in metric_series and q75_key in metric_series:
            q25_values = np.asarray(metric_series[q25_key], dtype=np.float64)
            q75_values = np.asarray(metric_series[q75_key], dtype=np.float64)
            mask = np.isfinite(q25_values) & np.isfinite(q75_values)
            if np.any(mask):
                x = np.asarray(steps, dtype=np.float64)[mask]
                low = q25_values[mask]
                high = q75_values[mask]
                axis.fill_between(x, low, high, alpha=0.20, color="tab:blue", label=f"{metric_name}_q25~q75")

        if finite_values.size > 0 and show_mean_line:
            mean_value = float(np.mean(finite_values))
            axis.axhline(mean_value, linestyle="--", linewidth=1.2, alpha=0.9, color="tab:red", label=f"mean={mean_value:.4g}")

        if finite_values.size > 0 and quantile_interval is not None:
            q_low, q_high = quantile_interval
            low_value, high_value = np.quantile(finite_values, [q_low, q_high])
            axis.axhspan(
                float(low_value),
                float(high_value),
                alpha=0.18,
                color="tab:orange",
                label=f"q{int(q_low*100)}-q{int(q_high*100)}",
            )

        if finite_values.size > 0 and quantile_lines is not None:
            for q in quantile_lines:
                q_value = float(np.quantile(finite_values, q))
                axis.axhline(
                    q_value,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.9,
                    color="tab:green",
                    label=f"q{int(q*100)}={q_value:.4g}",
                )

        if reward_reference_lines and metric_name in {"reward_sum", "return"}:
            for ref_name, ref_value in reward_reference_lines.items():
                axis.axhline(
                    float(ref_value),
                    linestyle="-.",
                    linewidth=1.0,
                    alpha=0.95,
                    color="tab:purple",
                    label=f"{ref_name}={float(ref_value):.4g}",
                )

        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("step")
    figure.suptitle(csv_file.stem)
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    matplotlib.pyplot.close(figure)
    return output_file


def _parse_float(raw: str | None) -> float:
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return math.nan


def _ordered_base_metrics(metric_names: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in metric_names:
        base = _metric_base_name(name)
        if base in seen:
            continue
        seen.add(base)
        ordered.append(base)
    return ordered


def _metric_base_name(name: str) -> str:
    if name.endswith("_q25") or name.endswith("_q50") or name.endswith("_q75"):
        return name[:-4]
    return name
