from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model
from plot import plot_model

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

def train(task_kwargs: dict, dataset_id: str, model_id: str, model_kwargs: dict) -> None:
    start = task_kwargs.get("start", 0)

    dataset = make_dataset(dataset_id, device="cuda")
    model = make_model(model_id, dataset, device="cuda", **model_kwargs)
    task_id = _task_id(dataset.id, model.id)

    if start > 0:
        model.load(task_id, start)
        
    path = train_model(
        agent=model,
        dataset=dataset,
        task_id=task_id,
        **task_kwargs,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    dataset_ids = list(dict.fromkeys(dataset_id for _, dataset_id, _, _ in TASKS))
    for task_kwargs, dataset_id, model_id, model_kwargs in TASKS:
        train(task_kwargs, dataset_id, model_id, model_kwargs)
        plot_model(dataset_ids.index(dataset_id) + 1, dataset_id, model_id)
