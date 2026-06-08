from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import metric_path
from ice_offline.tools.plotter import plot as plot_metric_eval


SHOW = False


def plot(metrics_path: str, eval_paths: str, output_path: Path) -> None:
    metrics_file = Path(metrics_path)
    eval_file = Path(eval_paths)
    if not metrics_file.exists():
        print(f"skip missing: {metrics_path}")
        return
    if not eval_file.exists():
        print(f"skip missing: {eval_paths}")
        return
    plot_metric_eval(str(metrics_file), str(eval_file), str(output_path))
    print(f"saved: {output_path}")


def plot_agent(index: int, agent_id: str, dataset_cls) -> None:
    dataset = dataset_cls()
    metrics_path = metric_path(dataset.id, agent_id)
    returns_path = eval_path(dataset.id, agent_id)
    output_path = VIEW_ROOT / "plot" / agent_id / f"{index}. {dataset.id}.png"
    plot(str(metrics_path), str(returns_path), output_path)


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    for i, dataset_cls in enumerate(dataset_class_list, start=1):
        for agent_id in agent_id_list:
            print(f"dataset={dataset_cls().id}, agent={agent_id}")
            plot_agent(i, agent_id, dataset_cls)


if __name__ == "__main__":
    main([], [])
