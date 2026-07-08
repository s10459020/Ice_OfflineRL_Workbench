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
    "cql_gap_0p0",
    "cql_gap_0p5",
    "cql_gap_1p0",
    "cql_gap_1p5",
    "cql_gp_gap_0p0",
    "cql_gp_gap_0p5",
    "cql_gp_gap_1p0",
    "cql_gp_gap_1p5",
]


def plot_agent(dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    metrics_output_path = metric_path(task_id)
    returns_output_path, steps_output_path = cal_eval(task_id, "train")
    output_path = plot_path("train", dataset_id, agent_id)

    print(f"plot task={task_id}")
    plot([metrics_output_path], [returns_output_path, steps_output_path], output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for agent_id in AGENTS:
            plot_agent(dataset_id, agent_id)
