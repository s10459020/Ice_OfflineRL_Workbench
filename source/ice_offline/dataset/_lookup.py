from ice_offline.dataset._spec import BaseDataset


DATASET_LOOKUP: dict[str, BaseDataset] = {
    "halfcheetah_expert": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/expert-v0", env_id="HalfCheetah-v5"),
    "halfcheetah_medium": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/medium-v0", env_id="HalfCheetah-v5"),
    "halfcheetah_simple": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/simple-v0", env_id="HalfCheetah-v5"),
    "hopper_expert": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/expert-v0", env_id="Hopper-v5"),
    "hopper_medium": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/medium-v0", env_id="Hopper-v5"),
    "hopper_simple": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/simple-v0", env_id="Hopper-v5"),
    "invertedpendulum_expert": BaseDataset(dataset_name="inverted_pendulum", dataset_id="mujoco/invertedpendulum/expert-v0", env_id="InvertedPendulum-v5"),
    "onerooms8_fullobs_optimal": BaseDataset(dataset_name="onerooms8", dataset_id="minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0", env_id="BabyAI-OneRoomS8-v0"),
    "walker2d_expert": BaseDataset(dataset_name="walker2d", dataset_id="mujoco/walker2d/expert-v0", env_id="Walker2d-v5"),
    "walker2d_medium": BaseDataset(dataset_name="walker2d", dataset_id="mujoco/walker2d/medium-v0", env_id="Walker2d-v5"),
    "walker2d_simple": BaseDataset(dataset_name="walker2d", dataset_id="mujoco/walker2d/simple-v0", env_id="Walker2d-v5"),
}


def get_dataset(dataset_id: str) -> BaseDataset:
    if dataset_id == "onerooms8_fullobs_optimal":
        from ice_offline.dataset.oneroom_s8 import OneRoomS8Dataset

        return OneRoomS8Dataset()
    return DATASET_LOOKUP[dataset_id]
