from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model

TASKS = [
    ({"steps": 100_000}, "hopper_d4rl_medium", "scas_model", {}),
    ({"steps": 100_000}, "hopper_d4rl_hybrid", "scas_model", {}),
    ({"steps": 100_000}, "hopper_d4rl_expert", "scas_model", {}),
    ({"steps": 100_000}, "hopper_random", "scas_model", {}),
    ({"steps": 100_000}, "hopper_replay_medium", "scas_model", {}),
    ({"steps": 100_000}, "hopper_replay_expert", "scas_model", {}),
    ({"steps": 100_000}, "hopper_d4rl_medium", "sdc_model", {}),
    ({"steps": 100_000}, "hopper_d4rl_hybrid", "sdc_model", {}),
    ({"steps": 100_000}, "hopper_d4rl_expert", "sdc_model", {}),
    ({"steps": 100_000}, "hopper_random", "sdc_model", {}),
    ({"steps": 100_000}, "hopper_replay_medium", "sdc_model", {}),
    ({"steps": 100_000}, "hopper_replay_expert", "sdc_model", {}),
]

TASK_KWARGS = {
    # "start": 0,
    # "steps": 100_000,
    # "save_interval": 20_000,
    # "print_interval": 200,
}

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
]

MODELS = [
    # ("scas_model", {}),
    # ("sdc_model", {}),
]


def normalize_tasks() -> list[tuple[dict[str, object], str, str, dict[str, object]]]:
    if TASKS:
        return TASKS
    return [
        (
            dict(TASK_KWARGS),
            dataset_id,
            model_id,
            model_kwargs,
        )
        for dataset_id in DATASETS
        for model_id, model_kwargs in MODELS
    ]


def main() -> None:
    tasks = normalize_tasks()

    for task_kwargs, dataset_id, model_id, model_kwargs in tasks:
        dataset = make_dataset(dataset_id, device="cuda")
        model = make_model(model_id, dataset, device="cuda", **model_kwargs)

        task_id = _task_id(dataset.id, model.id)
        start = task_kwargs.get("start", 0)
        if start > 0:
            model.load(task_id, start)
        print(
            f"task={task_id}, dataset={dataset.id}, model={model.id}, "
            f"model_kwargs={model_kwargs}"
        )
        path = train_model(
            agent=model,
            dataset=dataset,
            task_id=task_id,
            **task_kwargs,
        )
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
