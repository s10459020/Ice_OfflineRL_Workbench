from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ice_offline.tools.plotter import plot_csv
from ice_offline.config.paths import VIEW_ROOT, eval_path


SHOW = False


def plot(path: str, output_path: Path, *, show: bool = False) -> None:
    eval_file = Path(path)
    if not eval_file.exists():
        print(f"skip missing: {path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_csv(csv_paths=str(eval_file), plot_name=path, show=show, output_path=str(output_path))
    print(f"saved: {output_path}")


def plot_agent(index: int, agent_id: str, dataset_cls) -> None:
    dataset = dataset_cls()
    path = eval_path(dataset.id, agent_id)
    output_path = VIEW_ROOT / "evals" / agent_id / f"{index}. {dataset.id}.png"
    plot(str(path), output_path, show=SHOW)


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    for i, dataset_cls in enumerate(dataset_class_list, start=1):
        for agent_id in agent_id_list:
            print(f"dataset={dataset_cls().id}, agent={agent_id}")
            plot_agent(i, agent_id, dataset_cls)


if __name__ == "__main__":
    main([], [])
