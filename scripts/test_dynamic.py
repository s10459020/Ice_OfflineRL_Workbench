import argparse
import csv
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

DEFAULT_OUTPUT = table_path("dynamic", "success.csv")
TARGET_SUCCESS_RATE = 0.9
OUTLIER_RATIO = 0.0


@dataclass
class DynamicResult:
    dataset_id: str
    model_id: str
    step: int
    status: str
    raw_count: int
    count: int
    outlier_count: int
    outlier_ratio: float
    target_success_rate: float
    success: int
    success_rate: float | None
    threshold: float | None
    true_d_pr50: float | None
    true_d_var: float | None
    true_d_mean: float | None
    true_d_p95: float | None
    pred_d_pr50: float | None
    pred_d_var: float | None
    pred_d_mean: float | None
    pred_d_p90: float | None
    pred_d_p95: float | None
    pred_raw_mse: float | None
    pred_raw_rmse: float | None


def _score_dynamic(
    dataset_id: str,
    experiment_id: str,
    model_id: str,
    step: int,
    device: str,
    batch_size: int,
    target_success_rate: float,
    outlier_ratio: float,
) -> DynamicResult:
    task_id = experiment_task_id(experiment_id, model_id, dataset_id)
    path = model_path(task_id, step)
    if not path.exists():
        return DynamicResult(
            dataset_id=dataset_id,
            model_id=model_id,
            step=step,
            status="missing_model",
            raw_count=0,
            count=0,
            outlier_count=0,
            outlier_ratio=outlier_ratio,
            target_success_rate=target_success_rate,
            success=0,
            success_rate=None,
            threshold=None,
            true_d_pr50=None,
            true_d_var=None,
            true_d_mean=None,
            true_d_p95=None,
            pred_d_pr50=None,
            pred_d_var=None,
            pred_d_mean=None,
            pred_d_p90=None,
            pred_d_p95=None,
            pred_raw_mse=None,
            pred_raw_rmse=None,
        )

    dataset = make_dataset(dataset_id, device=device)
    model = make_model(model_id, dataset, device=device)
    model.load(path)
    model.prepare()

    buffer = dataset.buffer
    count = int(buffer.observations.shape[0])
    state_std = buffer.next_observations.std(dim=0, unbiased=False).clamp_min(1e-6)
    true_d_parts: list[torch.Tensor] = []
    pred_d_parts: list[torch.Tensor] = []
    pred_raw_mse_parts: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            observations = buffer.observations[start:end]
            actions = buffer.actions[start:end]
            next_observations = buffer.next_observations[start:end]
            prediction = model.forward(observations, actions)
            error = prediction - next_observations
            pred_raw_mse = error.square().mean(dim=1)
            true_d = ((next_observations - observations) / state_std).square().mean(dim=1).sqrt()
            pred_d = (error / state_std).square().mean(dim=1).sqrt()
            true_d_parts.append(true_d.detach().cpu())
            pred_d_parts.append(pred_d.detach().cpu())
            pred_raw_mse_parts.append(pred_raw_mse.detach().cpu())

    true_d_values = torch.cat(true_d_parts)
    pred_d_values = torch.cat(pred_d_parts)
    pred_raw_mse_values = torch.cat(pred_raw_mse_parts)

    raw_count = int(pred_d_values.shape[0])
    keep_count = int(raw_count * (1.0 - outlier_ratio))
    indices = torch.argsort(pred_d_values)[:keep_count]
    true_d_values = true_d_values[indices]
    pred_d_values = pred_d_values[indices]
    pred_raw_mse_values = pred_raw_mse_values[indices]

    threshold = float(torch.quantile(pred_d_values, target_success_rate).item())
    success = int((pred_d_values <= threshold).sum().item())
    count = int(pred_d_values.shape[0])
    pred_raw_mse = float(pred_raw_mse_values.mean().item())
    return DynamicResult(
        dataset_id=dataset_id,
        model_id=model_id,
        step=step,
        status="ok",
        raw_count=raw_count,
        count=count,
        outlier_count=raw_count - count,
        outlier_ratio=outlier_ratio,
        target_success_rate=target_success_rate,
        success=success,
        success_rate=success / count,
        threshold=threshold,
        true_d_pr50=float(torch.quantile(true_d_values, 0.50).item()),
        true_d_var=float(true_d_values.var(unbiased=False).item()),
        true_d_mean=float(true_d_values.mean().item()),
        true_d_p95=float(torch.quantile(true_d_values, 0.95).item()),
        pred_d_pr50=float(torch.quantile(pred_d_values, 0.50).item()),
        pred_d_var=float(pred_d_values.var(unbiased=False).item()),
        pred_d_mean=float(pred_d_values.mean().item()),
        pred_d_p90=float(torch.quantile(pred_d_values, 0.90).item()),
        pred_d_p95=float(torch.quantile(pred_d_values, 0.95).item()),
        pred_raw_mse=pred_raw_mse,
        pred_raw_rmse=pred_raw_mse ** 0.5,
    )


def _target_success_rate(config: dict[str, object], target_success_rate: float) -> float:
    if "target_success_rate" in config:
        return float(config["target_success_rate"])
    return target_success_rate


def _outlier_ratio(config: dict[str, object], outlier_ratio: float) -> float:
    if "outlier_ratio" in config:
        return float(config["outlier_ratio"])
    return outlier_ratio


def _write_results(path: Path, results: list[DynamicResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset",
            "model",
            "step",
            "status",
            "raw_count",
            "count",
            "outlier_count",
            "outlier_ratio",
            "target_success_rate",
            "success",
            "success_rate",
            "threshold",
            "true_d_pr50",
            "true_d_var",
            "true_d_mean",
            "true_d_p95",
            "pred_d_pr50",
            "pred_d_var",
            "pred_d_mean",
            "pred_d_p90",
            "pred_d_p95",
            "pred_raw_mse",
            "pred_raw_rmse",
        ])
        for result in results:
            writer.writerow([
                result.dataset_id,
                result.model_id,
                result.step,
                result.status,
                result.raw_count,
                result.count,
                result.outlier_count,
                _cell(result.outlier_ratio),
                _cell(result.target_success_rate),
                result.success,
                _cell(result.success_rate),
                _cell(result.threshold),
                _cell(result.true_d_pr50),
                _cell(result.true_d_var),
                _cell(result.true_d_mean),
                _cell(result.true_d_p95),
                _cell(result.pred_d_pr50),
                _cell(result.pred_d_var),
                _cell(result.pred_d_mean),
                _cell(result.pred_d_p90),
                _cell(result.pred_d_p95),
                _cell(result.pred_raw_mse),
                _cell(result.pred_raw_rmse),
            ])
    print(f"saved: {path}")


def _cell(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def _print_results(results: list[DynamicResult]) -> None:
    print("dataset,model,step,status,raw_count,count,outlier_count,outlier_ratio,target_success_rate,success,success_rate,threshold,true_d_pr50,true_d_var,pred_d_mean,pred_d_p90,pred_d_p95")
    for result in results:
        print(
            ",".join([
                result.dataset_id,
                result.model_id,
                str(result.step),
                result.status,
                str(result.raw_count),
                str(result.count),
                str(result.outlier_count),
                _cell(result.outlier_ratio),
                _cell(result.target_success_rate),
                str(result.success),
                _cell(result.success_rate),
                _cell(result.threshold),
                _cell(result.true_d_pr50),
                _cell(result.true_d_var),
                _cell(result.pred_d_mean),
                _cell(result.pred_d_p90),
                _cell(result.pred_d_p95),
            ])
        )


def main(default_output: Path = DEFAULT_OUTPUT, default_outlier_ratio: float = OUTLIER_RATIO) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="base_train")
    parser.add_argument("--target-success-rate", type=float, default=TARGET_SUCCESS_RATE)
    parser.add_argument("--outlier-ratio", type=float, default=default_outlier_ratio)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    results = [
        _score_dynamic(
            dataset_id=dataset_id,
            experiment_id=str(config.get("experiment", args.experiment)),
            model_id=model_id,
            step=step,
            device=args.device,
            batch_size=args.batch_size,
            target_success_rate=_target_success_rate(config, args.target_success_rate),
            outlier_ratio=_outlier_ratio(config, args.outlier_ratio),
        )
        for dataset_id, model_id, step, config in TASKS
    ]

    _print_results(results)
    _write_results(args.output, results)


if __name__ == "__main__":
    main()
