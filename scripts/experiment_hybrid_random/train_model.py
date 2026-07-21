from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import model_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model
from plot import plot_train

EXPERIMENT_TRAIN = "experience_hybrid_random_train"

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

MODELS = [
    ((0, 500_000), "scas_model"),
]

TASKS = []


def train(task_steps: tuple[int, int], dataset_id: str, model_id: str) -> str:
    id = experiment_task_id(EXPERIMENT_TRAIN, model_id, dataset_id)
    start, steps = task_steps
    dataset = make_dataset(dataset_id, device="cuda")
    model = make_model(model_id, dataset, device="cuda")
    if start > 0:
        model.load(model_path(id, start))
    path = train_model(
        agent=model,
        dataset=dataset,
        task_id=id,
        start=start,
        steps=steps,
    )
    print(f"saved: {path}")
    return id


if __name__ == "__main__":
    model_tasks = [
        (task_steps, dataset_id, model_id)
        for task_steps, model_id in MODELS
        for dataset_id in DATASETS
    ] + TASKS

    for task_steps, dataset_id, model_id in model_tasks:
        id = train(task_steps, dataset_id, model_id)
        plot_train(id, metric_path(id), [], dataset_id, model_id)
