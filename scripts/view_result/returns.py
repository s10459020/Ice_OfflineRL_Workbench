import json
from pathlib import Path

import h5py
import numpy as np

from ice_offline.dataset.loader.minari.loader import MinariLoader
from ice_offline.tools.paths import dataset_root
from view_result.skip import skip_missing_data
from view_result.task import is_d4rl_path
from view_result.task import source_path
from view_result.task import test_path


RETURNS_ROOT = Path("tmps/returns")


def returns_path(path: str) -> Path:
    return RETURNS_ROOT / f"{path.replace('/', '__')}.json"


def compute_minari_returns(path: str) -> list[float]:
    dataset = MinariLoader(dataset_root() / path)
    values = []
    for episode in dataset.load_episodes():
        values.append(float(episode.rewards.sum()))
    return values


def compute_d4rl_returns(path: str) -> list[float]:
    values = []
    current = 0.0
    with h5py.File(path, "r") as file:
        rewards = np.asarray(file["rewards"], dtype=np.float64)
        terminals = np.asarray(file["terminals"], dtype=np.bool_)
        timeouts = np.asarray(file["timeouts"], dtype=np.bool_)
    for reward, done in zip(rewards, np.logical_or(terminals, timeouts)):
        current += float(reward)
        if bool(done):
            values.append(current)
            current = 0.0
    return values


def compute_returns(path: str) -> list[float]:
    if is_d4rl_path(path):
        return compute_d4rl_returns(path)
    return compute_minari_returns(path)


def returns(path: str) -> list[float]:
    output_path = returns_path(path)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            return [float(value) for value in json.load(file)]

    values = compute_returns(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"saved: {output_path}")
    return values


def returns_list(dataset_class_list: list, agent_id_list: list[str]) -> list[str]:
    source_dataset_list = [source_path(dataset_cls) for dataset_cls in dataset_class_list]
    test_dataset_list = [
        test_path(dataset_cls, agent_id)
        for dataset_cls in dataset_class_list
        for agent_id in agent_id_list
    ]
    return [*source_dataset_list, *test_dataset_list]


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    for path in returns_list(dataset_class_list, agent_id_list):
        if skip_missing_data(path):
            continue
        print(f"returns={path}")
        returns(path)


if __name__ == "__main__":
    main([], [])
