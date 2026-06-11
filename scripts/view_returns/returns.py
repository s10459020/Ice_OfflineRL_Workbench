import json
from pathlib import Path

from ice_offline.dataset._spec import Dataset
from ice_offline.config.paths import data_path_test, returns_path


def returns(path: Path, episodes) -> None:
    values = [float(episode.rewards.sum()) for episode in episodes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"saved: {path}")


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    for dataset_cls in dataset_class_list:
        path = Path(dataset_cls().path)
        out_path = returns_path(dataset_cls().id)
        if not path.exists():
            print(f"skip missing: {path}")
            continue
        print(f"returns={path}")
        returns(out_path, dataset_cls().episodes)

    for dataset_cls in dataset_class_list:
        for agent_id in agent_id_list:
            path = data_path_test(dataset_cls().id, agent_id)
            out_path = returns_path(dataset_cls().id, agent_id)
            if not path.exists():
                print(f"skip missing: {path}")
                continue
            print(f"returns={path}")
            dataset = Dataset(path=path)
            returns(out_path, dataset.episodes)


if __name__ == "__main__":
    main([], [])


