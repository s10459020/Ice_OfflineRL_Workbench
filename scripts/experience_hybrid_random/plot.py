from ice_offline.config.paths import _task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.run.eval import cal_eval
from ice_offline.run.plot import plot

DATASETS = [
    "hopper_random_expert_3",
    "hopper_random_expert_5",
    "hopper_random_expert_7",
    "hopper_random_expert_9",
]

AGENTS = [
    "bc_deterministic",
    "bc_stochastic",
    "td3bc",
    "iql",
    "cql",
    "aspl",
    "sdc",
    "sdc_cql",
    "scas",
    "scaspl",
]

MODELS = [
    "scas_model",
    "sdc_model",
]


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
    for index, dataset_id in enumerate(DATASETS, start=1):
        for agent_id in AGENTS:
            plot_agent(index, dataset_id, agent_id)
        for model_id in MODELS:
            plot_model(index, dataset_id, model_id)
