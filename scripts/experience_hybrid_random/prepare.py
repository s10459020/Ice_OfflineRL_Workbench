from ice_offline.agent._lookup import make_model
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model as run_train_model
from config import DATASETS
from config import MODEL_TASKS
from plot import plot_model


def train_model(
    task_kwargs: dict,
    dataset_id: str,
    model_id: str,
    model_kwargs: dict,
) -> None:
    start = task_kwargs.get("start", 0)

    dataset = make_dataset(dataset_id, device="cuda")
    model = make_model(model_id, dataset, device="cuda", **model_kwargs)
    task_id = _task_id(dataset.id, model.id)

    if start > 0:
        model.load(task_id, start)

    path = run_train_model(
        agent=model,
        dataset=dataset,
        task_id=task_id,
        **task_kwargs,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    for task_kwargs, dataset_id, model_id, model_kwargs in MODEL_TASKS:
        train_model(task_kwargs, dataset_id, model_id, model_kwargs)
        plot_model(DATASETS.index(dataset_id) + 1, dataset_id, model_id)
