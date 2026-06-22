from pathlib import Path
from collections.abc import Callable

import gymnasium as gym

from ice_offline.config.paths import custom_dataset_path
from ice_offline.config.paths import DATASETS_ROOT
from ice_offline.dataset.base import Dataset
from ice_offline.dataset.d4rl import D4rlDataset
from ice_offline.dataset.minari import MinariDataset


DATASET_TABLE: dict[str, Callable[[str], Dataset]] = {
    "hopper_simple": lambda device: MinariDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("mujoco/hopper/simple-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/hopper/simple-v0", device=device),
    "hopper_medium": lambda device: MinariDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("mujoco/hopper/medium-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/hopper/medium-v0", device=device),
    "hopper_expert": lambda device: MinariDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("mujoco/hopper/expert-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/hopper/expert-v0", device=device),
    "hopper_one_simple": lambda device: Dataset(env_id="Hopper-v5", path=custom_dataset_path("hopper_one_simple"), device=device),
    "hopper_one_medium": lambda device: Dataset(env_id="Hopper-v5", path=custom_dataset_path("hopper_medium_one"), device=device),
    "hopper_one_expert": lambda device: Dataset(env_id="Hopper-v5", path=custom_dataset_path("hopper_expert_one"), device=device),
    "hopper_random": lambda device: D4rlDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("d4rl/hopper_random-v2/d4rl_data.hdf5"), device=device),
    "hopper_d4rl_medium": lambda device: D4rlDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("d4rl/hopper_medium-v2/d4rl_data.hdf5"), device=device),
    "hopper_d4rl_hybrid": lambda device: D4rlDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("d4rl/hopper_medium_expert-v2/d4rl_data.hdf5"), device=device),
    "hopper_d4rl_expert": lambda device: D4rlDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("d4rl/hopper_expert-v2/d4rl_data.hdf5"), device=device),
    "hopper_replay_expert": lambda device: D4rlDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("d4rl/hopper_full_replay-v2/d4rl_data.hdf5"), device=device),
    "hopper_replay_medium": lambda device: D4rlDataset(env_id="Hopper-v5", path=DATASETS_ROOT / Path("d4rl/hopper_medium_replay-v2/d4rl_data.hdf5"), device=device),
    "walker2d_simple": lambda device: MinariDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("mujoco/walker2d/simple-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/walker2d/simple-v0", device=device),
    "walker2d_medium": lambda device: MinariDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("mujoco/walker2d/medium-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/walker2d/medium-v0", device=device),
    "walker2d_expert": lambda device: MinariDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("mujoco/walker2d/expert-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/walker2d/expert-v0", device=device),
    "walker2d_random": lambda device: D4rlDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("d4rl/walker2d_random-v2/d4rl_data.hdf5"), device=device),
    "walker2d_d4rl_medium": lambda device: D4rlDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("d4rl/walker2d_medium-v2/d4rl_data.hdf5"), device=device),
    "walker2d_d4rl_hybrid": lambda device: D4rlDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("d4rl/walker2d_medium_expert-v2/d4rl_data.hdf5"), device=device),
    "walker2d_d4rl_expert": lambda device: D4rlDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("d4rl/walker2d_expert-v2/d4rl_data.hdf5"), device=device),
    "walker2d_replay_expert": lambda device: D4rlDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("d4rl/walker2d_full_replay-v2/d4rl_data.hdf5"), device=device),
    "walker2d_replay_medium": lambda device: D4rlDataset(env_id="Walker2d-v5", path=DATASETS_ROOT / Path("d4rl/walker2d_medium_replay-v2/d4rl_data.hdf5"), device=device),
    "halfcheetah_simple": lambda device: MinariDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("mujoco/halfcheetah/simple-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/halfcheetah/simple-v0", device=device),
    "halfcheetah_medium": lambda device: MinariDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("mujoco/halfcheetah/medium-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/halfcheetah/medium-v0", device=device),
    "halfcheetah_expert": lambda device: MinariDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("mujoco/halfcheetah/expert-v0/data/main_data.hdf5"), minari_dataset_id="mujoco/halfcheetah/expert-v0", device=device),
    "halfcheetah_random": lambda device: D4rlDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("d4rl/halfcheetah_random-v2/d4rl_data.hdf5"), device=device),
    "halfcheetah_d4rl_medium": lambda device: D4rlDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("d4rl/halfcheetah_medium-v2/d4rl_data.hdf5"), device=device),
    "halfcheetah_d4rl_hybrid": lambda device: D4rlDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("d4rl/halfcheetah_medium_expert-v2/d4rl_data.hdf5"), device=device),
    "halfcheetah_d4rl_expert": lambda device: D4rlDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("d4rl/halfcheetah_expert-v2/d4rl_data.hdf5"), device=device),
    "halfcheetah_replay_expert": lambda device: D4rlDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("d4rl/halfcheetah_full_replay-v2/d4rl_data.hdf5"), device=device),
    "halfcheetah_replay_medium": lambda device: D4rlDataset(env_id="HalfCheetah-v5", path=DATASETS_ROOT / Path("d4rl/halfcheetah_medium_replay-v2/d4rl_data.hdf5"), device=device),
}

def source_dataset_ids() -> list[str]:
    return list(DATASET_TABLE.keys())

def _lookup_env(id: str) -> Callable[[], gym.Env]:
    env_config: dict[str, Callable[[], gym.Env]] = {
        "hopper_simple": lambda: gym.make("Hopper-v5"),
        "hopper_medium": lambda: gym.make("Hopper-v5"),
        "hopper_expert": lambda: gym.make("Hopper-v5"),
        "hopper_d4rl_medium": lambda: gym.make("Hopper-v5"),
        "hopper_d4rl_hybrid": lambda: gym.make("Hopper-v5"),
        "hopper_d4rl_expert": lambda: gym.make("Hopper-v5"),
        "hopper_random": lambda: gym.make("Hopper-v5"),
        "hopper_replay_medium": lambda: gym.make("Hopper-v5"),
        "hopper_replay_expert": lambda: gym.make("Hopper-v5"),
        "hopper_one_simple": lambda: gym.make("Hopper-v5"),
        "hopper_one_medium": lambda: gym.make("Hopper-v5"),
        "hopper_one_expert": lambda: gym.make("Hopper-v5"),
        "walker2d_simple": lambda: gym.make("Walker2d-v5"),
        "walker2d_medium": lambda: gym.make("Walker2d-v5"),
        "walker2d_expert": lambda: gym.make("Walker2d-v5"),
        "walker2d_d4rl_medium": lambda: gym.make("Walker2d-v5"),
        "walker2d_d4rl_hybrid": lambda: gym.make("Walker2d-v5"),
        "walker2d_d4rl_expert": lambda: gym.make("Walker2d-v5"),
        "walker2d_random": lambda: gym.make("Walker2d-v5"),
        "walker2d_replay_expert": lambda: gym.make("Walker2d-v5"),
        "walker2d_replay_medium": lambda: gym.make("Walker2d-v5"),
        "halfcheetah_simple": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_medium": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_expert": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_d4rl_medium": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_d4rl_hybrid": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_d4rl_expert": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_random": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_replay_expert": lambda: gym.make("HalfCheetah-v5"),
        "halfcheetah_replay_medium": lambda: gym.make("HalfCheetah-v5"),
    }
    return env_config[id]


def make_dataset(id: str, device: str = "cuda") -> Dataset:
    dataset = DATASET_TABLE[id](device)
    dataset.id = id
    return dataset
