from ice_offline.agent._lookup import make_model
from ice_offline.dataset._lookup import make_dataset
from ice_offline.config.paths import _task_id
from ice_offline.run.train import train_model as run_train_model
from plot import plot_model

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
    # "hopper_random",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    "halfcheetah_d4rl_medium",
    "halfcheetah_d4rl_hybrid",
    "halfcheetah_d4rl_expert",
    "halfcheetah_random",
    "halfcheetah_replay_medium",
    "halfcheetah_replay_expert",
]

MODELS = [
    (100_000, "scas_model"),
    # (100_000, "sdc_model"),
]


def train_model(
    task_kwargs: dict,
    dataset_id: str,
    model_id: str,
) -> None:
    dataset = make_dataset(dataset_id, device="cuda")
    model = make_model(model_id, dataset, device="cuda")
    task_id = _task_id(dataset.id, model.id)

    path = run_train_model(
        agent=model,
        dataset=dataset,
        task_id=task_id,
        **task_kwargs,
    )
    print(f"saved: {path}")
    
if __name__ == "__main__":
    for index, dataset_id in enumerate(DATASETS, start=1):
        for steps, model_id in MODELS:
            train_model({"steps": steps}, dataset_id, model_id)
            plot_model(index, dataset_id, model_id)
