from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ice_offline.plot.plotter import plot_csv
from view_result.skip import skip_missing
from view_result.task import AGENT_ID_LIST
from view_result.task import DATASET_CLASS_LIST
from view_result.task import dataset_id
from view_result.task import plot_output_path


EVAL_ROOT = Path("tmps/eval")
SHOW = False


def plot(dataset_path: str, output_path: Path, *, show: bool = False) -> None:
    eval_dir = EVAL_ROOT / dataset_path
    if skip_missing(dataset_path, eval_dir):
        return
    csv_paths = [str(path) for path in sorted(eval_dir.glob("*.csv"))]
    if len(csv_paths) == 0:
        print(f"skip missing csv: {eval_dir}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_csv(csv_paths=csv_paths, plot_name=dataset_path, show=show, output_path=str(output_path))
    print(f"saved: {output_path}")


def plot_agent(index: int, agent_id: str, dataset_cls) -> None:
    id_ = dataset_id(dataset_cls)
    dataset_path = f"{id_}-{agent_id}-v0"
    output_path = plot_output_path(index, dataset_cls, agent_id)
    plot(dataset_path, output_path, show=SHOW)


def main() -> None:
    for i, dataset_cls in enumerate(DATASET_CLASS_LIST, start=1):
        for agent_id in AGENT_ID_LIST:
            print(f"dataset={dataset_id(dataset_cls)}, agent={agent_id}")
            plot_agent(i, agent_id, dataset_cls)


if __name__ == "__main__":
    main()
