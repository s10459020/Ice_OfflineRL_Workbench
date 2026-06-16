from ice_offline.config.paths import _task_id
from ice_offline.config.paths import data_path_train
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.run.eval import cal_returns
from ice_offline.run.eval import cal_steps
from ice_offline.run.plot import plot


TASKS = [
    ({"steps": 200_000}, "hopper_medium", {}, "aspl", {"alpha": 0.5}),
]


def view_train(index: int, dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    input_path = data_path_train(task_id)
    returns_output_path = returns_path("train", task_id)
    steps_output_path = steps_path("train", task_id)
    metrics_output_path = metric_path(task_id)
    output_path = plot_path(index, dataset_id, agent_id)

    print(f"plot dataset={dataset_id}, agent={agent_id}")
    cal_returns(input_path, returns_output_path)
    cal_steps(input_path, steps_output_path)
    plot([metrics_output_path], [returns_output_path, steps_output_path], output_path)
    print(f"saved: {output_path}")


def main() -> None:
    for index, (_, dataset_id, _, agent_id, _) in enumerate(TASKS, start=1):
        view_train(index, dataset_id, agent_id)


if __name__ == "__main__":
    main()
