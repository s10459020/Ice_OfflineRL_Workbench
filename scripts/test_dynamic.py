import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import model_path
from ice_offline.config.paths import table_path
from ice_offline.dataset._lookup import make_dataset


EXPERIMENT_TRAIN = "base_train"
TRAINING_STEP = 500_000
DEVICE = "cpu"
STD_EPSILON = 1e-6
DENOMINATOR_EPSILON = 1e-12

DEFAULT_OUTPUT = table_path("dynamic", "global_standardized_transition_relative_rmse.csv")
DEFAULT_TABLE_OUTPUT = table_path("dynamic", "env_dataset_type.csv")

DATASET_TYPES = (
    ("d4rl_medium", "medium"),
    ("d4rl_expert", "expert"),
    ("d4rl_hybrid", "hybrid"),
    ("replay_medium", "medium_replay"),
    ("replay_expert", "expert_replay"),
)

TABLE_ENVIRONMENTS = (
    "Hopper",
    "Walker2D",
    "HalfCheetah",
    "Walker2D(ORIGIN)",
)

TASKS = [
    # format: (dataset_id, model_name, training_step, config)
    # Walker2D uses normalization_dynamic by default.
    *[
        (
            f"hopper_{dataset_key}",
            "scas_model",
            TRAINING_STEP,
            {"environment": "Hopper", "dataset": dataset_name},
        )
        for dataset_key, dataset_name in DATASET_TYPES
    ],
    *[
        (
            f"walker2d_{dataset_key}",
            "normalization_dynamic",
            TRAINING_STEP,
            {"environment": "Walker2D", "dataset": dataset_name},
        )
        for dataset_key, dataset_name in DATASET_TYPES
    ],
    *[
        (
            f"halfcheetah_{dataset_key}",
            "scas_model",
            TRAINING_STEP,
            {"environment": "HalfCheetah", "dataset": dataset_name},
        )
        for dataset_key, dataset_name in DATASET_TYPES
    ],
    *[
        (
            f"walker2d_{dataset_key}",
            "scas_model",
            TRAINING_STEP,
            {"environment": "Walker2D(ORIGIN)", "dataset": dataset_name},
        )
        for dataset_key, dataset_name in DATASET_TYPES
    ],
]


@dataclass
class DynamicResult:
    environment: str
    dataset: str
    model_name: str
    training_step: int
    sample_count: int
    state_dimension: int
    global_standardized_transition_relative_rmse: float
    percentage_relative_error: float


def _task_parts(task: tuple) -> tuple[str, str, int, dict[str, object]]:
    dataset_id, model_name, training_step, config = task
    return dataset_id, model_name, training_step, config


def _default_environment(dataset_id: str) -> str:
    environment_id = dataset_id.split("_", 1)[0]
    names = {
        "hopper": "Hopper",
        "walker2d": "Walker2D",
        "halfcheetah": "HalfCheetah",
    }
    return names[environment_id]


def _default_dataset_type(dataset_id: str) -> str:
    for dataset_key, dataset_name in DATASET_TYPES:
        if dataset_id.endswith(dataset_key):
            return dataset_name
    return dataset_id


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def _model_checkpoint(experiment_id: str, model_name: str, dataset_id: str, training_step: int) -> Path:
    task_id = experiment_task_id(experiment_id, model_name, dataset_id)
    return model_path(task_id, training_step)


def _verify_metric(
    prediction_squared_sum: float,
    transition_squared_sum: float,
    value_count: int,
    metric: float,
    percentage: float,
    denominator_epsilon: float,
) -> None:
    if not math.isfinite(prediction_squared_sum):
        raise ValueError("prediction_squared_sum is not finite")
    if not math.isfinite(transition_squared_sum):
        raise ValueError("transition_squared_sum is not finite")
    if transition_squared_sum <= 0.0:
        raise ValueError("transition_squared_sum must be positive")
    if not math.isfinite(metric):
        raise ValueError("global_standardized_transition_relative_rmse is not finite")
    if not math.isfinite(percentage):
        raise ValueError("percentage_relative_error is not finite")
    prediction_rmse = math.sqrt(prediction_squared_sum / value_count)
    transition_rmse = math.sqrt(transition_squared_sum / value_count)
    ratio_metric = prediction_rmse / transition_rmse
    if not math.isclose(metric, ratio_metric, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("sum-of-squares metric and RMSE-ratio metric differ")
    if not math.isclose(percentage, metric * 100.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("percentage_relative_error must equal metric * 100")
    if denominator_epsilon >= transition_squared_sum:
        raise ValueError("denominator_epsilon materially changes the transition denominator")


def _evaluate_task(
    dataset_id: str,
    model_name: str,
    training_step: int,
    config: dict[str, object],
    experiment_id: str,
    device: str,
    std_epsilon: float,
    denominator_epsilon: float,
) -> DynamicResult | None:
    path = _model_checkpoint(experiment_id, model_name, dataset_id, training_step)
    if not path.exists():
        print(f"missing_model: {path}", file=sys.stderr)
        return None

    dataset = make_dataset(dataset_id, device=device)
    model = make_model(model_name, dataset, device=device)
    model.load(path)
    model.prepare()

    buffer = dataset.buffer
    observations = buffer.observations
    actions = buffer.actions
    next_observations = buffer.next_observations
    sample_count = int(observations.shape[0])
    state_dimension = int(observations.shape[1])
    safe_state_std = observations.std(dim=0, unbiased=False).clamp_min(std_epsilon)

    with torch.inference_mode():
        predicted_next_state = model.forward(observations, actions)
        standardized_prediction_error = (predicted_next_state - next_observations) / safe_state_std
        standardized_true_transition = (next_observations - observations) / safe_state_std
        prediction_squared_sum = float(standardized_prediction_error.double().square().sum().item())
        transition_squared_sum = float(standardized_true_transition.double().square().sum().item())

    value_count = sample_count * state_dimension
    metric = math.sqrt(prediction_squared_sum / max(transition_squared_sum, denominator_epsilon))
    percentage = metric * 100.0
    _verify_metric(
        prediction_squared_sum=prediction_squared_sum,
        transition_squared_sum=transition_squared_sum,
        value_count=value_count,
        metric=metric,
        percentage=percentage,
        denominator_epsilon=denominator_epsilon,
    )

    return DynamicResult(
        environment=str(config.get("environment", _default_environment(dataset_id))),
        dataset=str(config.get("dataset", _default_dataset_type(dataset_id))),
        model_name=model_name,
        training_step=training_step,
        sample_count=sample_count,
        state_dimension=state_dimension,
        global_standardized_transition_relative_rmse=metric,
        percentage_relative_error=percentage,
    )


def _write_results(path: Path, results: list[DynamicResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "environment",
            "dataset",
            "model_name",
            "training_step",
            "sample_count",
            "state_dimension",
            "global_standardized_transition_relative_rmse",
            "percentage_relative_error",
        ])
        for result in results:
            writer.writerow([
                result.environment,
                result.dataset,
                result.model_name,
                result.training_step,
                result.sample_count,
                result.state_dimension,
                _format_float(result.global_standardized_transition_relative_rmse),
                f"{result.percentage_relative_error:.2f}",
            ])
    print(f"saved: {path}")


def _write_table(path: Path, results: list[DynamicResult]) -> None:
    values = {
        (result.environment, result.dataset): result.percentage_relative_error
        for result in results
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["environment", *[dataset_name for _, dataset_name in DATASET_TYPES]])
        for environment in TABLE_ENVIRONMENTS:
            writer.writerow([
                environment,
                *[
                    "" if (environment, dataset_name) not in values else f"{values[(environment, dataset_name)]:.2f}"
                    for _, dataset_name in DATASET_TYPES
                ],
            ])
    print(f"saved: {path}")


def _print_results(results: list[DynamicResult]) -> None:
    print("environment,dataset,model_name,training_step,sample_count,state_dimension,global_standardized_transition_relative_rmse,percentage_relative_error")
    for result in results:
        print(
            ",".join([
                result.environment,
                result.dataset,
                result.model_name,
                str(result.training_step),
                str(result.sample_count),
                str(result.state_dimension),
                _format_float(result.global_standardized_transition_relative_rmse),
                f"{result.percentage_relative_error:.2f}",
            ])
        )


def _print_tasks(experiment_id: str) -> None:
    for task in TASKS:
        dataset_id, model_name, training_step, config = _task_parts(task)
        path = _model_checkpoint(experiment_id, model_name, dataset_id, training_step)
        print(
            ",".join([
                str(config.get("environment", _default_environment(dataset_id))),
                str(config.get("dataset", _default_dataset_type(dataset_id))),
                dataset_id,
                model_name,
                str(training_step),
                "ok" if path.exists() else "missing_model",
            ])
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=EXPERIMENT_TRAIN)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--std-epsilon", type=float, default=STD_EPSILON)
    parser.add_argument("--denominator-epsilon", type=float, default=DENOMINATOR_EPSILON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        _print_tasks(args.experiment)
        return

    results: list[DynamicResult] = []
    for task in TASKS:
        dataset_id, model_name, training_step, config = _task_parts(task)
        result = _evaluate_task(
            dataset_id=dataset_id,
            model_name=model_name,
            training_step=training_step,
            config=config,
            experiment_id=args.experiment,
            device=args.device,
            std_epsilon=args.std_epsilon,
            denominator_epsilon=args.denominator_epsilon,
        )
        if result is not None:
            results.append(result)

    _print_results(results)
    _write_results(args.output, results)
    _write_table(args.table_output, results)


if __name__ == "__main__":
    main()
