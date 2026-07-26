import argparse
import csv
import math
from pathlib import Path

import torch

from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import model_path
from ice_offline.config.paths import table_path
from ice_offline.dataset._lookup import make_dataset
from test_dynamic import TASKS


EXPERIMENT_TRAIN = "base_train"
DEVICE = "cpu"
BATCH_SIZE = 32_768
EPSILON = 1e-6
OUTPUT_GROUP = "dynamic/diagnostics"

PERCENTILES = (10, 25, 50, 75, 90, 95, 99)
LOW_MOVEMENT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
MOVEMENT_GROUPS = (
    (0.00, 0.10, "p00_p10"),
    (0.10, 0.25, "p10_p25"),
    (0.25, 0.50, "p25_p50"),
    (0.50, 0.75, "p50_p75"),
    (0.75, 0.90, "p75_p90"),
    (0.90, 1.00, "p90_p100"),
)

DIMENSION_NAMES = {
    "hopper": (
        "torso height rootz position",
        "torso angle rooty position",
        "thigh angle",
        "leg angle",
        "foot angle",
        "torso x velocity",
        "torso z velocity",
        "torso angular velocity",
        "thigh angular velocity",
        "leg angular velocity",
        "foot angular velocity",
    ),
    "walker2d": (
        "torso height rootz position",
        "torso angle rooty position",
        "right thigh angle",
        "right leg angle",
        "right foot angle",
        "left thigh angle",
        "left leg angle",
        "left foot angle",
        "torso x velocity",
        "torso z velocity",
        "torso angular velocity",
        "right thigh angular velocity",
        "right leg angular velocity",
        "right foot angular velocity",
        "left thigh angular velocity",
        "left leg angular velocity",
        "left foot angular velocity",
    ),
    "halfcheetah": (
        "front tip height rootz position",
        "front tip angle rooty position",
        "back thigh angle",
        "back shin angle",
        "back foot angle",
        "front thigh angle",
        "front shin angle",
        "front foot angle",
        "front tip x velocity",
        "front tip z velocity",
        "front tip angular velocity",
        "back thigh angular velocity",
        "back shin angular velocity",
        "back foot angular velocity",
        "front thigh angular velocity",
        "front shin angular velocity",
        "front foot angular velocity",
    ),
}


def _environment_id(dataset_id: str) -> str:
    return dataset_id.split("_", 1)[0]


def _task_parts(task: tuple) -> tuple[str, str, str, int, dict]:
    if len(task) == 5:
        train_dataset_id, validation_dataset_id, model_id, step, config = task
        return train_dataset_id, validation_dataset_id, model_id, step, config
    dataset_id, model_id, step, config = task
    validation_dataset_id = str(config.get("validation_dataset_id", config.get("validation_dataset", dataset_id)))
    return dataset_id, validation_dataset_id, model_id, step, config


def _format(value: int | float | str | bool | None) -> int | str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _nearest_rank(values: torch.Tensor, percentile: float) -> float:
    sorted_values = values.sort().values
    value_count = int(sorted_values.shape[0])
    rank = math.ceil((percentile / 100.0) * value_count)
    rank = max(1, min(rank, value_count))
    return float(sorted_values[rank - 1].item())


def _percentile_columns(prefix: str, values: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}_p{percentile}": _nearest_rank(values, percentile)
        for percentile in PERCENTILES
    }


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = centered_x.square().sum().sqrt() * centered_y.square().sum().sqrt()
    return float((centered_x * centered_y).sum().div(denominator.clamp_min(EPSILON)).item())


def _score_task(
    train_dataset_id: str,
    validation_dataset_id: str,
    model_id: str,
    step: int,
    device: str,
    batch_size: int,
    epsilon: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    task_id = experiment_task_id(EXPERIMENT_TRAIN, model_id, train_dataset_id)
    path = model_path(task_id, step)
    environment_id = _environment_id(train_dataset_id)

    if not path.exists():
        return {
            "environment": environment_id,
            "train_dataset": train_dataset_id,
            "validation_dataset": validation_dataset_id,
            "model": model_id,
            "step": step,
            "status": "missing_model",
        }, [], []

    train_dataset = make_dataset(train_dataset_id, device=device)
    validation_dataset = make_dataset(validation_dataset_id, device=device)
    model = make_model(model_id, train_dataset, device=device)
    model.load(path)
    model.prepare()

    train_buffer = train_dataset.buffer
    validation_buffer = validation_dataset.buffer
    state_std = train_buffer.observations.std(dim=0, unbiased=False).clamp_min(epsilon)
    observations = validation_buffer.observations
    actions = validation_buffer.actions
    next_observations = validation_buffer.next_observations
    sample_count = int(observations.shape[0])
    dimension_count = int(observations.shape[1])

    true_d_parts = []
    pred_d_parts = []
    relative_parts = []
    norm_error_abs_sum = 0.0
    norm_error_sq_sum = 0.0
    transition_sq_sum = torch.zeros(dimension_count, dtype=torch.float64)
    prediction_error_sq_sum = torch.zeros(dimension_count, dtype=torch.float64)

    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            o = observations[start:end]
            a = actions[start:end]
            on = next_observations[start:end]
            prediction = model.forward(o, a)
            error = prediction - on
            transition = on - o
            normalized_error = error / state_std

            true_d = (transition / state_std).square().mean(dim=1).sqrt()
            pred_d = normalized_error.square().mean(dim=1).sqrt()
            relative_error = pred_d / true_d.clamp_min(epsilon)

            true_d_parts.append(true_d.cpu())
            pred_d_parts.append(pred_d.cpu())
            relative_parts.append(relative_error.cpu())
            norm_error_abs_sum += float(normalized_error.abs().sum().item())
            norm_error_sq_sum += float(normalized_error.square().sum().item())
            transition_sq_sum += transition.square().sum(dim=0).double().cpu()
            prediction_error_sq_sum += error.square().sum(dim=0).double().cpu()

    true_d_values = torch.cat(true_d_parts)
    pred_d_values = torch.cat(pred_d_parts)
    relative_values = torch.cat(relative_parts)
    state_std_cpu = state_std.cpu()
    sorted_true_indices = true_d_values.argsort()
    normalized_value_count = sample_count * dimension_count

    low_counts = {}
    for threshold in LOW_MOVEMENT_THRESHOLDS:
        count = int((true_d_values < threshold).sum().item())
        key = str(threshold).replace(".", "_")
        low_counts[f"true_d_lt_{key}_count"] = count
        low_counts[f"true_d_lt_{key}_pct"] = 100.0 * count / sample_count

    summary = {
        "environment": environment_id,
        "train_dataset": train_dataset_id,
        "validation_dataset": validation_dataset_id,
        "model": model_id,
        "step": step,
        "status": "ok",
        "validation_independent": validation_dataset_id != train_dataset_id,
        "validation_samples": sample_count,
        "state_dimensions": dimension_count,
        "denominator_epsilon": epsilon,
        "state_std_min": float(state_std_cpu.min().item()),
        "state_std_median": _nearest_rank(state_std_cpu, 50.0),
        "state_std_max": float(state_std_cpu.max().item()),
        "mean_normalized_prediction_error": float(pred_d_values.mean().item()),
        "global_normalized_rmse": math.sqrt(norm_error_sq_sum / normalized_value_count),
        "global_normalized_mae": norm_error_abs_sum / normalized_value_count,
        "true_d_relative_corr": _correlation(true_d_values, relative_values),
        **low_counts,
        **_percentile_columns("true_d", true_d_values),
        **_percentile_columns("pred_d", pred_d_values),
        **_percentile_columns("relative_error", relative_values),
    }

    group_rows = []
    for lower, upper, group_name in MOVEMENT_GROUPS:
        start = int(math.floor(lower * sample_count))
        end = sample_count if upper == 1.0 else int(math.floor(upper * sample_count))
        indices = sorted_true_indices[start:end]
        group_true = true_d_values[indices]
        group_pred = pred_d_values[indices]
        group_relative = relative_values[indices]
        group_rows.append({
            "environment": environment_id,
            "train_dataset": train_dataset_id,
            "validation_dataset": validation_dataset_id,
            "model": model_id,
            "step": step,
            "group": group_name,
            "sample_count": int(indices.shape[0]),
            "mean_true_d": float(group_true.mean().item()),
            "mean_pred_d": float(group_pred.mean().item()),
            "pred_d_p90": _nearest_rank(group_pred, 90.0),
            "relative_error_median": _nearest_rank(group_relative, 50.0),
            "relative_error_p90": _nearest_rank(group_relative, 90.0),
        })

    transition_rms = (transition_sq_sum / sample_count).sqrt()
    prediction_error_rms = (prediction_error_sq_sum / sample_count).sqrt()
    dimension_ratio = prediction_error_rms / transition_rms.clamp_min(epsilon)
    dimension_names = DIMENSION_NAMES.get(environment_id, ())
    dimension_rows = []
    for rank, dim in enumerate(dimension_ratio.argsort(descending=True).tolist(), start=1):
        name = dimension_names[dim] if dim < len(dimension_names) else f"dim_{dim}"
        dimension_rows.append({
            "environment": environment_id,
            "train_dataset": train_dataset_id,
            "validation_dataset": validation_dataset_id,
            "model": model_id,
            "step": step,
            "rank": rank,
            "dimension": dim,
            "dimension_name": name,
            "state_std": float(state_std_cpu[dim].item()),
            "transition_rms": float(transition_rms[dim].item()),
            "prediction_error_rms": float(prediction_error_rms[dim].item()),
            "dimension_rms_ratio": float(dimension_ratio[dim].item()),
        })

    return summary, group_rows, dimension_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--scas-only", action="store_true")
    args = parser.parse_args()

    tasks = [
        task
        for task in TASKS
        if not args.scas_only or _task_parts(task)[2] == "scas_model"
    ]

    summary_rows = []
    group_rows = []
    dimension_rows = []

    for task in tasks:
        train_dataset_id, validation_dataset_id, model_id, step, config = _task_parts(task)
        summary, groups, dimensions = _score_task(
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            model_id=model_id,
            step=step,
            device=args.device,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
        )
        summary_rows.append(summary)
        group_rows.extend(groups)
        dimension_rows.extend(dimensions)
        print(
            f"{summary['status']}: {train_dataset_id}, {model_id}, "
            f"step={step}, samples={summary.get('validation_samples', 0)}"
        )

    _write_csv(table_path(OUTPUT_GROUP, "summary.csv"), summary_rows)
    _write_csv(table_path(OUTPUT_GROUP, "movement_groups.csv"), group_rows)
    _write_csv(table_path(OUTPUT_GROUP, "dimensions.csv"), dimension_rows)
    print(f"saved: {table_path(OUTPUT_GROUP, 'summary.csv')}")
    print(f"saved: {table_path(OUTPUT_GROUP, 'movement_groups.csv')}")
    print(f"saved: {table_path(OUTPUT_GROUP, 'dimensions.csv')}")


if __name__ == "__main__":
    main()
