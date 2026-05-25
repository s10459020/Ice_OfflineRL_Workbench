import csv
from pathlib import Path

import matplotlib.pyplot
import numpy as np


RUNNER_ID = "hopper_simple_bc_deterministic"
EVAL_DIR = Path("tmps/eval") / RUNNER_ID
PLOT_DIR = Path("tmps/eval/plots")
METRICS = ["loss_pi", "reward_sum"]


def _load_eval_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_stream:
        reader = csv.DictReader(csv_stream)
        if not reader.fieldnames:
            raise ValueError(f"csv has no header: {csv_path}")

        step_key = None
        sample_keys: list[str] = []
        for key in reader.fieldnames:
            key_stripped = str(key).strip().strip('"').lstrip("\ufeff")
            if key_stripped == "step":
                step_key = key
            else:
                sample_keys.append(key)

        if step_key is None:
            raise ValueError(f"csv has no step column: {csv_path}")
        if not sample_keys:
            raise ValueError(f"csv has no sample columns: {csv_path}")

        steps: list[float] = []
        samples: list[list[float]] = []
        for row in reader:
            step_raw = row.get(step_key)
            if step_raw is None or step_raw == "":
                continue
            steps.append(float(step_raw))

            values: list[float] = []
            for sample_key in sample_keys:
                raw = row.get(sample_key)
                values.append(float(raw) if raw not in (None, "") else np.nan)
            samples.append(values)

    return np.asarray(steps, dtype=np.float64), np.asarray(samples, dtype=np.float64)


if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = matplotlib.pyplot.subplots(
        nrows=len(METRICS),
        ncols=1,
        figsize=(11, 4.2 * len(METRICS)),
        sharex=True,
    )
    if len(METRICS) == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, METRICS):
        csv_path = EVAL_DIR / f"{metric_name}.csv"
        steps, sample_matrix = _load_eval_csv(csv_path)

        q25 = np.nanquantile(sample_matrix, 0.25, axis=1)
        q50 = np.nanquantile(sample_matrix, 0.50, axis=1)
        q75 = np.nanquantile(sample_matrix, 0.75, axis=1)

        axis.plot(steps, q50, linewidth=1.8, label=f"{metric_name}_q50")
        axis.fill_between(steps, q25, q75, alpha=0.22, label=f"{metric_name}_q25~q75")
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("step")
    figure.suptitle(RUNNER_ID)
    figure.tight_layout()

    output_path = PLOT_DIR / f"{RUNNER_ID}__bc_eval_q2575.png"
    figure.savefig(output_path, dpi=150)
    matplotlib.pyplot.close(figure)

    print(f"saved: {output_path}")
