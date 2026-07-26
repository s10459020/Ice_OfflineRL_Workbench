import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import model_path
from ice_offline.config.paths import table_path
from ice_offline.dataset._lookup import make_dataset


TASKS = [
    ("hopper_d4rl_medium", "scas_model", 500_000, {}),
    ("hopper_d4rl_expert", "scas_model", 500_000, {}),
    ("hopper_d4rl_hybrid", "scas_model", 500_000, {}),
    ("hopper_replay_medium", "scas_model", 500_000, {}),
    ("hopper_replay_expert", "scas_model", 500_000, {}),
    ("walker2d_d4rl_medium", "scas_model", 500_000, {}),
    ("walker2d_d4rl_expert", "scas_model", 500_000, {}),
    ("walker2d_d4rl_hybrid", "scas_model", 500_000, {}),
    ("walker2d_replay_medium", "scas_model", 500_000, {}),
    ("walker2d_replay_expert", "scas_model", 500_000, {}),
    ("walker2d_d4rl_medium", "normalization_dynamic", 500_000, {}),
    ("walker2d_d4rl_expert", "normalization_dynamic", 500_000, {}),
    ("walker2d_d4rl_hybrid", "normalization_dynamic", 500_000, {}),
    ("walker2d_replay_medium", "normalization_dynamic", 500_000, {}),
    ("walker2d_replay_expert", "normalization_dynamic", 500_000, {}),
    ("halfcheetah_d4rl_medium", "scas_model", 500_000, {}),
    ("halfcheetah_d4rl_expert", "scas_model", 500_000, {}),
    ("halfcheetah_d4rl_hybrid", "scas_model", 500_000, {}),
    ("halfcheetah_replay_medium", "scas_model", 500_000, {}),
    ("halfcheetah_replay_expert", "scas_model", 500_000, {}),
]

DEFAULT_OUTPUT = table_path("dynamic", "relative_threshold.csv")
TARGET_COVERAGE = 90.0
OUTLIER_RATIO = 0.0
EPSILON = 1e-6


@dataclass
class DynamicResult:
    train_dataset_id: str
    validation_dataset_id: str
    model_id: str
    step: int
    status: str
    raw_validation_sample_count: int
    validation_sample_count: int
    outlier_count: int
    outlier_ratio: float
    target_coverage: float
    actual_coverage: float | None
    relative_error_threshold: float | None
    percentage_error_threshold: float | None
    relative_error_mean: float | None
    relative_error_median: float | None
    relative_error_p90: float | None
    relative_error_p95: float | None
    relative_error_p99: float | None
    d_true_under_epsilon_count: int
    raw_d_true_under_epsilon_count: int
    true_d_pr50: float | None
    true_d_mean: float | None
    pred_d_pr50: float | None
    pred_d_mean: float | None
    validation_independent: bool


def _nearest_rank(values: torch.Tensor, coverage: float) -> float:
    sorted_values = values.sort().values
    value_count = int(sorted_values.shape[0])
    rank = math.ceil((coverage / 100.0) * value_count)
    rank = max(1, min(rank, value_count))
    return float(sorted_values[rank - 1].item())


def _score_dynamic(
    train_dataset_id: str,
    validation_dataset_id: str,
    experiment_id: str,
    model_id: str,
    step: int,
    device: str,
    batch_size: int,
    target_coverage: float,
    outlier_ratio: float,
    epsilon: float,
) -> DynamicResult:
    task_id = experiment_task_id(experiment_id, model_id, train_dataset_id)
    path = model_path(task_id, step)
    if not path.exists():
        return DynamicResult(
            train_dataset_id=train_dataset_id,
            validation_dataset_id=validation_dataset_id,
            model_id=model_id,
            step=step,
            status="missing_model",
            raw_validation_sample_count=0,
            validation_sample_count=0,
            outlier_count=0,
            outlier_ratio=outlier_ratio,
            target_coverage=target_coverage,
            actual_coverage=None,
            relative_error_threshold=None,
            percentage_error_threshold=None,
            relative_error_mean=None,
            relative_error_median=None,
            relative_error_p90=None,
            relative_error_p95=None,
            relative_error_p99=None,
            d_true_under_epsilon_count=0,
            raw_d_true_under_epsilon_count=0,
            true_d_pr50=None,
            true_d_mean=None,
            pred_d_pr50=None,
            pred_d_mean=None,
            validation_independent=validation_dataset_id != train_dataset_id,
        )

    train_dataset = make_dataset(train_dataset_id, device=device)
    validation_dataset = make_dataset(validation_dataset_id, device=device)
    model = make_model(model_id, train_dataset, device=device)
    model.load(path)
    model.prepare()

    train_buffer = train_dataset.buffer
    validation_buffer = validation_dataset.buffer
    state_std = train_buffer.observations.std(dim=0, unbiased=False).clamp_min(epsilon)

    buffer = validation_buffer
    count = int(buffer.observations.shape[0])
    true_d_parts: list[torch.Tensor] = []
    pred_d_parts: list[torch.Tensor] = []
    relative_error_parts: list[torch.Tensor] = []
    d_true_under_epsilon_parts: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            observations = buffer.observations[start:end]
            actions = buffer.actions[start:end]
            next_observations = buffer.next_observations[start:end]
            prediction = model.forward(observations, actions)
            error = prediction - next_observations
            true_d = ((next_observations - observations) / state_std).square().mean(dim=1).sqrt()
            pred_d = (error / state_std).square().mean(dim=1).sqrt()
            relative_error = pred_d / true_d.clamp_min(epsilon)
            d_true_under_epsilon = true_d < epsilon
            true_d_parts.append(true_d.detach().cpu())
            pred_d_parts.append(pred_d.detach().cpu())
            relative_error_parts.append(relative_error.detach().cpu())
            d_true_under_epsilon_parts.append(d_true_under_epsilon.detach().cpu())

    true_d_values = torch.cat(true_d_parts)
    pred_d_values = torch.cat(pred_d_parts)
    relative_error_values = torch.cat(relative_error_parts)
    d_true_under_epsilon_values = torch.cat(d_true_under_epsilon_parts)

    raw_count = int(relative_error_values.shape[0])
    raw_d_true_under_epsilon_count = int(d_true_under_epsilon_values.sum().item())
    keep_count = int(raw_count * (1.0 - outlier_ratio))
    indices = torch.argsort(relative_error_values)[:keep_count]
    true_d_values = true_d_values[indices]
    pred_d_values = pred_d_values[indices]
    relative_error_values = relative_error_values[indices]
    d_true_under_epsilon_values = d_true_under_epsilon_values[indices]

    threshold = _nearest_rank(relative_error_values, target_coverage)
    success = int((relative_error_values <= threshold).sum().item())
    count = int(relative_error_values.shape[0])
    return DynamicResult(
        train_dataset_id=train_dataset_id,
        validation_dataset_id=validation_dataset_id,
        model_id=model_id,
        step=step,
        status="ok",
        raw_validation_sample_count=raw_count,
        validation_sample_count=count,
        outlier_count=raw_count - count,
        outlier_ratio=outlier_ratio,
        target_coverage=target_coverage,
        actual_coverage=100.0 * success / count,
        relative_error_threshold=threshold,
        percentage_error_threshold=threshold * 100.0,
        relative_error_mean=float(relative_error_values.mean().item()),
        relative_error_median=_nearest_rank(relative_error_values, 50.0),
        relative_error_p90=_nearest_rank(relative_error_values, 90.0),
        relative_error_p95=_nearest_rank(relative_error_values, 95.0),
        relative_error_p99=_nearest_rank(relative_error_values, 99.0),
        d_true_under_epsilon_count=int(d_true_under_epsilon_values.sum().item()),
        raw_d_true_under_epsilon_count=raw_d_true_under_epsilon_count,
        true_d_pr50=_nearest_rank(true_d_values, 50.0),
        true_d_mean=float(true_d_values.mean().item()),
        pred_d_pr50=_nearest_rank(pred_d_values, 50.0),
        pred_d_mean=float(pred_d_values.mean().item()),
        validation_independent=validation_dataset_id != train_dataset_id,
    )


def _target_coverage(config: dict[str, object], target_coverage: float) -> float:
    if "target_coverage" in config:
        return float(config["target_coverage"])
    if "target_success_rate" in config:
        return float(config["target_success_rate"]) * 100.0
    return target_coverage


def _outlier_ratio(config: dict[str, object], outlier_ratio: float) -> float:
    if "outlier_ratio" in config:
        return float(config["outlier_ratio"])
    return outlier_ratio


def _task_parts(task):
    if len(task) == 5:
        train_dataset_id, validation_dataset_id, model_id, step, config = task
        return train_dataset_id, validation_dataset_id, model_id, step, config
    dataset_id, model_id, step, config = task
    validation_dataset_id = str(config.get("validation_dataset_id", config.get("validation_dataset", dataset_id)))
    return dataset_id, validation_dataset_id, model_id, step, config


def _write_results(path: Path, results: list[DynamicResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "train_dataset",
            "validation_dataset",
            "model",
            "step",
            "status",
            "raw_validation_sample_count",
            "validation_sample_count",
            "outlier_count",
            "outlier_ratio",
            "target_coverage",
            "relative_error_threshold",
            "percentage_error_threshold",
            "actual_coverage",
            "relative_error_mean",
            "relative_error_median",
            "relative_error_p90",
            "relative_error_p95",
            "relative_error_p99",
            "d_true_under_epsilon_count",
            "raw_d_true_under_epsilon_count",
            "true_d_pr50",
            "true_d_mean",
            "pred_d_pr50",
            "pred_d_mean",
            "validation_independent",
        ])
        for result in results:
            writer.writerow([
                result.train_dataset_id,
                result.validation_dataset_id,
                result.model_id,
                result.step,
                result.status,
                result.raw_validation_sample_count,
                result.validation_sample_count,
                result.outlier_count,
                _cell(result.outlier_ratio),
                _cell(result.target_coverage),
                _cell(result.relative_error_threshold),
                _cell(result.percentage_error_threshold),
                _cell(result.actual_coverage),
                _cell(result.relative_error_mean),
                _cell(result.relative_error_median),
                _cell(result.relative_error_p90),
                _cell(result.relative_error_p95),
                _cell(result.relative_error_p99),
                result.d_true_under_epsilon_count,
                result.raw_d_true_under_epsilon_count,
                _cell(result.true_d_pr50),
                _cell(result.true_d_mean),
                _cell(result.pred_d_pr50),
                _cell(result.pred_d_mean),
                result.validation_independent,
            ])
    print(f"saved: {path}")


def _cell(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def _print_results(results: list[DynamicResult]) -> None:
    print("train_dataset,validation_dataset,model,step,status,samples,target_coverage,percentage_error_threshold,actual_coverage,mean_relative_error,median_relative_error,relative_error_p90,relative_error_p95,relative_error_p99,d_true_under_epsilon_count,validation_independent")
    for result in results:
        print(
            ",".join([
                result.train_dataset_id,
                result.validation_dataset_id,
                result.model_id,
                str(result.step),
                result.status,
                str(result.validation_sample_count),
                _cell(result.target_coverage),
                _cell(result.percentage_error_threshold),
                _cell(result.actual_coverage),
                _cell(result.relative_error_mean),
                _cell(result.relative_error_median),
                _cell(result.relative_error_p90),
                _cell(result.relative_error_p95),
                _cell(result.relative_error_p99),
                str(result.d_true_under_epsilon_count),
                str(result.validation_independent),
            ])
        )


def main(default_output: Path = DEFAULT_OUTPUT, default_outlier_ratio: float = OUTLIER_RATIO) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="base_train")
    parser.add_argument("--target-coverage", type=float, default=TARGET_COVERAGE)
    parser.add_argument("--target-success-rate", type=float, default=None)
    parser.add_argument("--outlier-ratio", type=float, default=default_outlier_ratio)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    target_coverage = args.target_coverage
    if args.target_success_rate is not None:
        target_coverage = args.target_success_rate * 100.0

    results: list[DynamicResult] = []
    for task in TASKS:
        train_dataset_id, validation_dataset_id, model_id, step, config = _task_parts(task)
        results.append(
            _score_dynamic(
                train_dataset_id=train_dataset_id,
                validation_dataset_id=validation_dataset_id,
                experiment_id=str(config.get("experiment", args.experiment)),
                model_id=model_id,
                step=step,
                device=args.device,
                batch_size=args.batch_size,
                target_coverage=_target_coverage(config, target_coverage),
                outlier_ratio=_outlier_ratio(config, args.outlier_ratio),
                epsilon=float(config.get("epsilon", args.epsilon)),
            )
        )

    _print_results(results)
    _write_results(args.output, results)


if __name__ == "__main__":
    main()
