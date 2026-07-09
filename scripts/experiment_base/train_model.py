from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model
from plot import plot_train

EXPERIMENT = "base"
EXPERIMENT_TRAIN = "base_train"

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
    # "hopper_random",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_random",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_d4rl_expert",
    # "halfcheetah_random",
    # "halfcheetah_replay_medium",
    # "halfcheetah_replay_expert",
]

MODELS = [
    (100_000, "scas_model"),
]


def train(steps: int, dataset_id: str, model_id: str) -> str:
    id = task_id(dataset_id, model_id, EXPERIMENT_TRAIN)
    dataset = make_dataset(dataset_id, device="cuda")
    model = make_model(model_id, dataset, device="cuda")
    path = train_model(
        agent=model,
        dataset=dataset,
        task_id=id,
        steps=steps,
    )
    print(f"saved: {path}")
    return id


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for steps, model_id in MODELS:
            id = train(steps, dataset_id, model_id)
            plot_train(id, metric_path(id), [])
