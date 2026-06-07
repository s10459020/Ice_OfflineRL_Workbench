import csv
import json
from pathlib import Path

from ice_offline.dataset.halfcheetah_random import HalfCheetahRandomDataset
from ice_offline.dataset.hopper_random import HopperRandomDataset
from ice_offline.dataset.walker2d_random import Walker2dRandomDataset
from ice_offline.config.paths import DATASETS_ROOT, VIEW_ROOT, returns_path


RANDOM_DATASET_CLASS_BY_ENV_NAME = {
    "hopper": HopperRandomDataset,
    "halfcheetah": HalfCheetahRandomDataset,
    "walker2d": Walker2dRandomDataset,
}


def bottom_path(dataset_cls) -> Path:
    if Path(dataset_cls().path).relative_to(DATASETS_ROOT).parts[0] == "d4rl":
        random_dataset_cls = RANDOM_DATASET_CLASS_BY_ENV_NAME[dataset_cls().id.split("_", 1)[0]]
        return returns_path(random_dataset_cls().id)
    return returns_path(dataset_cls().id, "random")


def top_path(dataset_cls) -> Path | None:
    if dataset_cls is RANDOM_DATASET_CLASS_BY_ENV_NAME[dataset_cls().id.split("_", 1)[0]]:
        return None
    return returns_path(dataset_cls().id)


def read_values(path: Path) -> list[float] | None:
    if not path:
        return None
    if not path.exists():
        print(f"skip missing: {path}")
        return None
    with path.open("r", encoding="utf-8") as file:
        return [float(value) for value in json.load(file)]


def mean_return(path: Path | None) -> float | None:
    values = read_values(path)
    if values is None:
        return None
    if not values:
        return None
    return sum(values) / len(values)


def max_return(path: Path | None) -> float | None:
    values = read_values(path)
    if values is None:
        return None
    if not values:
        return None
    return max(values)


def cell(value: float | None) -> str:
    if value is None:
        return ""
    return str(float(value))


def scale(value: float | None, bottom: float | None, top: float | None) -> float | None:
    if value is None or bottom is None or top is None:
        return None
    if top == bottom:
        return None
    return (value - bottom) / (top - bottom) * 100.0


def actual_row(dataset_cls, agent_list: list[str]) -> list[str]:
    env_id = dataset_cls().env_id
    group_name = dataset_cls().id
    values = [mean_return(bottom_path(dataset_cls))]
    for agent_id in agent_list:
        values.append(mean_return(returns_path(dataset_cls().id, agent_id)))
    values.append(mean_return(top_path(dataset_cls)))
    return [env_id, group_name, *[cell(value) for value in values]]


def normalized_row(
    dataset_cls,
    agent_list: list[str],
    *,
    use_max_top: bool,
) -> list[str]:
    env_id = dataset_cls().env_id
    group_name = dataset_cls().id
    bottom = mean_return(bottom_path(dataset_cls))
    top = max_return(top_path(dataset_cls)) if use_max_top else mean_return(top_path(dataset_cls))
    values = []
    for agent_id in agent_list:
        values.append(scale(mean_return(returns_path(dataset_cls().id, agent_id)), bottom, top))
    return [env_id, group_name, *[cell(value) for value in values]]


def save_csv(name: str, header: list[str], rows: list[list[str]]) -> Path:
    out_path = VIEW_ROOT / "returns" / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved: {out_path}")
    return out_path


def actual_rows(dataset_cls_list: list, agent_list: list[str]) -> list[list[str]]:
    rows = []
    for dataset_cls in dataset_cls_list:
        rows.append(actual_row(dataset_cls, agent_list))
    return rows


def normalized_rows(dataset_cls_list: list, agent_list: list[str], *, use_max_top: bool) -> list[list[str]]:
    rows = []
    for dataset_cls in dataset_cls_list:
        rows.append(normalized_row(dataset_cls, agent_list, use_max_top=use_max_top))
    return rows


def dataset_classes_by_env(dataset_class_list: list) -> dict[str, list]:
    grouped = {}
    for dataset_cls in dataset_class_list:
        grouped.setdefault(dataset_cls().env_id, []).append(dataset_cls)
    return grouped


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    agent_list = [agent_id for agent_id in agent_id_list if agent_id != "random"]
    save_csv("actual_returns.csv", ["env", "task", "random", *agent_list, "dataset"], actual_rows(dataset_class_list, agent_list))
    save_csv("mean_normalized_returns.csv", ["env", "task", *agent_list], normalized_rows(dataset_class_list, agent_list, use_max_top=False))
    save_csv("max_normalized_returns.csv", ["env", "task", *agent_list], normalized_rows(dataset_class_list, agent_list, use_max_top=True))


if __name__ == "__main__":
    main([], [])


