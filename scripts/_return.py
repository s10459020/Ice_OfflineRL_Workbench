import json
from pathlib import Path

import h5py
import numpy as np

from ice_offline.data.minari.loader import MinariLoader
from ice_offline.tools.paths import minari_root


RETURNS_ROOT = Path("tmps/returns")

RETURNS_LIST = [
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_full_replay-v2.hdf5",
    "tmps/datasets/d4rl/hopper_expert-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium_replay-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium_expert-v2.hdf5",
    "mujoco/hopper/simple-v0/data/main_data.hdf5",
    "mujoco/hopper/medium-v0/data/main_data.hdf5",
    "mujoco/hopper/expert-v0/data/main_data.hdf5",
    "test/hopper_random-random-v0/data/main_data.hdf5",
    "test/hopper_random-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_random-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_random-td3bc-v0/data/main_data.hdf5",
    "test/hopper_random-iql-v0/data/main_data.hdf5",
    "test/hopper_random-cql-v0/data/main_data.hdf5",
    "test/hopper_random-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_random-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_random-aspl-v0/data/main_data.hdf5",
    "test/hopper_random-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_random-scas_min-v0/data/main_data.hdf5",
    "test/hopper_replay-random-v0/data/main_data.hdf5",
    "test/hopper_replay-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_replay-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_replay-td3bc-v0/data/main_data.hdf5",
    "test/hopper_replay-iql-v0/data/main_data.hdf5",
    "test/hopper_replay-cql-v0/data/main_data.hdf5",
    "test/hopper_replay-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_replay-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_replay-aspl-v0/data/main_data.hdf5",
    "test/hopper_replay-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_replay-scas_min-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-random-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-td3bc-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-iql-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-cql-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-aspl-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_expert_d4rl-scas_min-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-random-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-td3bc-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-iql-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-cql-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-aspl-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_medium_d4rl-scas_min-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-random-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-td3bc-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-iql-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-cql-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-aspl-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-scas_min-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-random-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-td3bc-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-iql-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-cql-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-aspl-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-scas_min-v0/data/main_data.hdf5",
    "test/hopper_simple-random-v0/data/main_data.hdf5",
    "test/hopper_simple-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_simple-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_simple-td3bc-v0/data/main_data.hdf5",
    "test/hopper_simple-iql-v0/data/main_data.hdf5",
    "test/hopper_simple-cql-v0/data/main_data.hdf5",
    "test/hopper_simple-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_simple-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_simple-aspl-v0/data/main_data.hdf5",
    "test/hopper_simple-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_simple-scas_min-v0/data/main_data.hdf5",
    "test/hopper_medium-random-v0/data/main_data.hdf5",
    "test/hopper_medium-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_medium-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_medium-td3bc-v0/data/main_data.hdf5",
    "test/hopper_medium-iql-v0/data/main_data.hdf5",
    "test/hopper_medium-cql-v0/data/main_data.hdf5",
    "test/hopper_medium-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_medium-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_medium-aspl-v0/data/main_data.hdf5",
    "test/hopper_medium-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_medium-scas_min-v0/data/main_data.hdf5",
    "test/hopper_expert-random-v0/data/main_data.hdf5",
    "test/hopper_expert-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_expert-bc_stochastic-v0/data/main_data.hdf5",
    "test/hopper_expert-td3bc-v0/data/main_data.hdf5",
    "test/hopper_expert-iql-v0/data/main_data.hdf5",
    "test/hopper_expert-cql-v0/data/main_data.hdf5",
    "test/hopper_expert-cql_max_q-v0/data/main_data.hdf5",
    "test/hopper_expert-cql_soft_q-v0/data/main_data.hdf5",
    "test/hopper_expert-aspl-v0/data/main_data.hdf5",
    "test/hopper_expert-scas_mean-v0/data/main_data.hdf5",
    "test/hopper_expert-scas_min-v0/data/main_data.hdf5",
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
    if dataset_path.startswith("tmps/datasets/d4rl/"):
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


if __name__ == "__main__":
    for dataset_path in RETURNS_LIST:
        data_path = Path(dataset_path)
        if not data_path.exists():
            data_path = minari_root() / dataset_path
        if not data_path.exists():
            print(f"skip missing: {dataset_path}")
            continue
        print(f"returns={dataset_path}")
        returns(dataset_path)
