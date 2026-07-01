from collections.abc import Callable

import gymnasium as gym

from ice_offline.config.paths import custom_dataset_path
from ice_offline.config.paths import d4rl_dataset_path
from ice_offline.config.paths import minari_dataset_path
from ice_offline.dataset.base import Dataset
from ice_offline.dataset.d4rl import D4rlDataset
from ice_offline.dataset.hybrid import HybridDataset
from ice_offline.dataset.minari import MinariDataset


def _minari_dataset(env_id: str, minari_dataset_id: str) -> Callable[[str], Dataset]:
    return lambda device: MinariDataset(
        env_id=env_id,
        path=minari_dataset_path(minari_dataset_id),
        minari_dataset_id=minari_dataset_id,
        device=device,
    )


def _d4rl_dataset(env_id: str, d4rl_dataset_id: str) -> Callable[[str], Dataset]:
    return lambda device: D4rlDataset(
        env_id=env_id,
        path=d4rl_dataset_path(d4rl_dataset_id),
        device=device,
    )


def _custom_dataset(env_id: str, dataset_id: str) -> Callable[[str], Dataset]:
    return lambda device: Dataset(
        env_id=env_id,
        path=custom_dataset_path(dataset_id),
        device=device,
    )


def _hybrid_dataset(
    dataset_id: str,
    env_id: str,
    dataset_a_id: str,
    dataset_b_id: str,
    count: int,
    random_ratio: float,
) -> Callable[[str], Dataset]:
    def make_hybrid_dataset(device: str) -> Dataset:
        path = custom_dataset_path(dataset_id)
        if path.exists():
            return Dataset(env_id=env_id, path=path, device=device)

        dataset_a = make_dataset(dataset_a_id, device="cpu")
        dataset_b = make_dataset(dataset_b_id, device="cpu")
        hybrid_dataset = HybridDataset(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            sample_count=count,
            random_ratio=random_ratio,
            device=device,
            env_id=env_id,
        )
        hybrid_dataset.save(path=path, dataset_id=dataset_id)
        hybrid_dataset.path = path
        return hybrid_dataset

    return make_hybrid_dataset


DATASET_TABLE: dict[str, Callable[[str], Dataset]] = {
    "hopper_simple": _minari_dataset("Hopper-v5", "mujoco/hopper/simple-v0"),
    "hopper_medium": _minari_dataset("Hopper-v5", "mujoco/hopper/medium-v0"),
    "hopper_expert": _minari_dataset("Hopper-v5", "mujoco/hopper/expert-v0"),
    "hopper_d4rl_medium": _d4rl_dataset("Hopper-v5", "hopper_medium-v2"),
    "hopper_d4rl_hybrid": _d4rl_dataset("Hopper-v5", "hopper_medium_expert-v2"),
    "hopper_d4rl_expert": _d4rl_dataset("Hopper-v5", "hopper_expert-v2"),
    "hopper_replay_expert": _d4rl_dataset("Hopper-v5", "hopper_full_replay-v2"),
    "hopper_replay_medium": _d4rl_dataset("Hopper-v5", "hopper_medium_replay-v2"),
    "hopper_random": _d4rl_dataset("Hopper-v5", "hopper_random-v2"),
    "hopper_random_expert_3": _hybrid_dataset("hopper_random_expert_3", "Hopper-v5", "hopper_random", "hopper_d4rl_expert", 1_000_000, 0.3),
    "hopper_random_expert_5": _hybrid_dataset("hopper_random_expert_5", "Hopper-v5", "hopper_random", "hopper_d4rl_expert", 1_000_000, 0.5),
    "hopper_random_expert_7": _hybrid_dataset("hopper_random_expert_7", "Hopper-v5", "hopper_random", "hopper_d4rl_expert", 1_000_000, 0.7),
    "hopper_random_expert_9": _hybrid_dataset("hopper_random_expert_9", "Hopper-v5", "hopper_random", "hopper_d4rl_expert", 1_000_000, 0.9),
    "hopper_road_medium": _custom_dataset("Hopper-v5", "hopper_road_medium"),
    "hopper_road_expert": _custom_dataset("Hopper-v5", "hopper_road_expert"),
    "walker2d_simple": _minari_dataset("Walker2d-v5", "mujoco/walker2d/simple-v0"),
    "walker2d_medium": _minari_dataset("Walker2d-v5", "mujoco/walker2d/medium-v0"),
    "walker2d_expert": _minari_dataset("Walker2d-v5", "mujoco/walker2d/expert-v0"),
    "walker2d_random": _d4rl_dataset("Walker2d-v5", "walker2d_random-v2"),
    "walker2d_d4rl_medium": _d4rl_dataset("Walker2d-v5", "walker2d_medium-v2"),
    "walker2d_d4rl_hybrid": _d4rl_dataset("Walker2d-v5", "walker2d_medium_expert-v2"),
    "walker2d_d4rl_expert": _d4rl_dataset("Walker2d-v5", "walker2d_expert-v2"),
    "walker2d_replay_expert": _d4rl_dataset("Walker2d-v5", "walker2d_full_replay-v2"),
    "walker2d_replay_medium": _d4rl_dataset("Walker2d-v5", "walker2d_medium_replay-v2"),
    "halfcheetah_simple": _minari_dataset("HalfCheetah-v5", "mujoco/halfcheetah/simple-v0"),
    "halfcheetah_medium": _minari_dataset("HalfCheetah-v5", "mujoco/halfcheetah/medium-v0"),
    "halfcheetah_expert": _minari_dataset("HalfCheetah-v5", "mujoco/halfcheetah/expert-v0"),
    "halfcheetah_random": _d4rl_dataset("HalfCheetah-v5", "halfcheetah_random-v2"),
    "halfcheetah_d4rl_medium": _d4rl_dataset("HalfCheetah-v5", "halfcheetah_medium-v2"),
    "halfcheetah_d4rl_hybrid": _d4rl_dataset("HalfCheetah-v5", "halfcheetah_medium_expert-v2"),
    "halfcheetah_d4rl_expert": _d4rl_dataset("HalfCheetah-v5", "halfcheetah_expert-v2"),
    "halfcheetah_replay_expert": _d4rl_dataset("HalfCheetah-v5", "halfcheetah_full_replay-v2"),
    "halfcheetah_replay_medium": _d4rl_dataset("HalfCheetah-v5", "halfcheetah_medium_replay-v2"),
}

def source_dataset_ids() -> list[str]:
    return list(DATASET_TABLE.keys())

def make_dataset(id: str, device: str = "cuda") -> Dataset:
    dataset = DATASET_TABLE[id](device)
    dataset.id = id
    return dataset
