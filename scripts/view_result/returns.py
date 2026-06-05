import json
from pathlib import Path

from ice_offline.dataset._spec import Dataset
from ice_offline.tools.paths import dataset_root
from view_result.utils import skip_missing


RETURNS_ROOT = Path("tmps/returns")


def returns(path: Path, episodes) -> None:
    values = [float(episode.rewards.sum()) for episode in episodes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"saved: {path}")


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    for dataset_cls in dataset_class_list:
        task_id = f"{dataset_cls().id}-v0"
        path = Path(dataset_cls().path).relative_to(dataset_root())
        out_path = RETURNS_ROOT / f"{task_id}.json"
        if skip_missing(path):
            continue
        print(f"returns={path}")
        returns(out_path, dataset_cls().episodes)

    for dataset_cls in dataset_class_list:
        for agent_id in agent_id_list:
            task_id = f"{dataset_cls().id}-{agent_id}-v0"
            path = Path("test") / task_id / "data" / "main_data.hdf5"
            out_path = RETURNS_ROOT / f"{task_id}.json"
            if skip_missing(path):
                continue
            print(f"returns={path}")
            dataset = Dataset(path=dataset_root() / path)
            returns(out_path, dataset.episodes)


if __name__ == "__main__":
    main([], [])
