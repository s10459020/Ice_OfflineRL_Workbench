from dataclasses import dataclass
from pathlib import Path

from ice_offline.data.d4rl.loader import D4rlLoader
from ice_offline.dataset._spec import Dataset


@dataclass
class HopperMediumReplayDataset(Dataset):
    id: str = "hopper_medium_replay_d4rl"
    env_id: str = "Hopper-v5"
    dataset_path: str | Path = Path("tmps/datasets/d4rl/hopper_medium_replay-v2.hdf5")

    def load(self, dataset_path: str | Path | None = None):
        return super().load(dataset_path or self.dataset_path)

    def make_loader(self):
        return D4rlLoader(self.dataset_path, device=self.device)
