from ice_offline.config.paths import _task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.run.eval import cal_eval
from ice_offline.run.plot import plot

DATASETS = [
    "hopper_replay_medium",
    "hopper_replay_expert",
]

AGENTS = [
    "td3",
    "td3_n",
    "td3_gp",
    "td3bc",
    "td3bc_n",
    "td3bc_gp",
]


def plot_agent(dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    metrics_output_path = metric_path(task_id)
    returns_output_path, steps_output_path = cal_eval(task_id, "train")
    output_path = plot_path(dataset_id, agent_id)

    print(f"plot dataset={dataset_id}, agent={agent_id}")
    plot([metrics_output_path], [returns_output_path, steps_output_path], output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for agent_id in AGENTS:
            plot_agent(dataset_id, agent_id)
