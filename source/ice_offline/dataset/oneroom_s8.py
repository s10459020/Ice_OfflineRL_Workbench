import gymnasium as gym
import minigrid # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.dataset._spec import BaseDataset
from ice_offline.env.common.mission_text_wrapper import MissionTextWrapper
from ice_offline.env.common.no_jpeg_image_wrapper import NoJpegImageWrapper


class OneRoomS8Dataset(BaseDataset):
    dataset_name = "onerooms8"
    dataset_id = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
    env_id = "BabyAI-OneRoomS8-v0"

    def __init__(self) -> None:
        super().__init__(
            dataset_name=self.dataset_name,
            dataset_id=self.dataset_id,
            env_id=self.env_id,
        )

    def obs_encode_batch(self, o_batch: dict[str, np.ndarray]) -> np.ndarray:
        imgs = np.asarray(o_batch["image"], dtype=np.float32) # [B, H, W, C]
        return imgs.reshape(imgs.shape[0], -1)                # [B, D]

    def obs_encode(self, o: dict[str, np.ndarray]) -> np.ndarray:
        imgs = np.asarray(o["image"], dtype=np.float32)       # [H, W, C]
        return imgs.reshape(-1)                               # [D]

    def make_eval_env(self):
        env = gym.make(self.env_id)
        env = FullyObsWrapper(env)
        return env  

    def make_collect_env(self) -> gym.Env:
        env = gym.make(self.env_id)
        env = FullyObsWrapper(env)
        env = MissionTextWrapper(env)
        env = NoJpegImageWrapper(env)
        return env
