import json
from pathlib import Path

import h5py
import numpy as np

from ice_offline.data.minari.loader import MinariLoader
from ice_offline.tools.paths import minari_root
from view_result.skip import skip_missing_data
from view_result.task import AGENT_ID_LIST
from view_result.task import DATASET_CLASS_LIST
from view_result.task import is_d4rl_dataset_path
from view_result.task import source_dataset_path
from view_result.task import test_dataset_path


RETURNS_ROOT = Path("tmps/returns")

SOURCE_DATASET_LIST = [source_dataset_path(dataset_cls) for dataset_cls in DATASET_CLASS_LIST]

RETURNS_LIST = [
    *SOURCE_DATASET_LIST,
    *[
        test_dataset_path(dataset_cls, agent_id)
        for dataset_cls in DATASET_CLASS_LIST
        for agent_id in AGENT_ID_LIST
    ],
]


def returns_path(dataset_path: str) -> Path:
    return RETURNS_ROOT / f"{dataset_path.replace('/', '__')}.json"


def compute_minari_returns(dataset_path: str) -> list[float]:
    dataset = MinariLoader(minari_root() / dataset_path)
    values = []
    for episode in dataset.iterate_episodes():
        values.append(float(episode.rewards.sum()))
    return values


def compute_d4rl_returns(dataset_path: str) -> list[float]:
    values = []
    current = 0.0
    with h5py.File(dataset_path, "r") as file:
        rewards = np.asarray(file["rewards"], dtype=np.float64)
        terminals = np.asarray(file["terminals"], dtype=np.bool_)
        timeouts = np.asarray(file["timeouts"], dtype=np.bool_)
    for reward, done in zip(rewards, np.logical_or(terminals, timeouts)):
        current += float(reward)
        if bool(done):
            values.append(current)
            current = 0.0
    return values


def compute_returns(dataset_path: str) -> list[float]:
    if is_d4rl_dataset_path(dataset_path):
        return compute_d4rl_returns(dataset_path)
    return compute_minari_returns(dataset_path)


def returns(dataset_path: str) -> list[float]:
    path = returns_path(dataset_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return [float(value) for value in json.load(file)]

    values = compute_returns(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"saved: {path}")
    return values


def main() -> None:
    for dataset_path in RETURNS_LIST:
        if skip_missing_data(dataset_path):
            continue
        print(f"returns={dataset_path}")
        returns(dataset_path)


if __name__ == "__main__":
    main()
