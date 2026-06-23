from ice_offline.config.paths import _task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.run.eval import cal_eval
from ice_offline.run.plot import plot
from config import AGENT_TASKS
from config import DATASETS
from config import MODEL_TASKS


def plot_item(index: int, dataset_id: str, id: str, eval_output_paths: list[str]) -> None:
    task_id = _task_id(dataset_id, id)
    metrics_output_path = metric_path(task_id)
    output_path = plot_path(index, dataset_id, id)

    print(f"plot dataset={dataset_id}, id={id}")
    plot([metrics_output_path], eval_output_paths, output_path)
    print(f"saved: {output_path}")


def plot_agent(index: int, dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    returns_output_path, steps_output_path = cal_eval(task_id, "train")
    plot_item(index, dataset_id, agent_id, [returns_output_path, steps_output_path])


def plot_model(index: int, dataset_id: str, model_id: str) -> None:
    plot_item(index, dataset_id, model_id, [])


if __name__ == "__main__":
    for _, dataset_id, agent_id, _ in AGENT_TASKS:
        plot_agent(DATASETS.index(dataset_id) + 1, dataset_id, agent_id)
    for _, dataset_id, model_id, _ in MODEL_TASKS:
        plot_model(DATASETS.index(dataset_id) + 1, dataset_id, model_id)
