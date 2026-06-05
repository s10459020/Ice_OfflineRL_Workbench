from dataclasses import dataclass
from pathlib import Path

from ice_offline.data.d4rl.loader import D4rlLoader
from ice_offline.dataset._spec import Dataset


@dataclass
class Walker2dReplayDataset(Dataset):
    id: str = "walker2d_replay"
    env_id: str = "Walker2d-v5"
    dataset_path: str | Path = Path("tmps/datasets/d4rl/walker2d_full_replay-v2.hdf5")

    def load(self, dataset_path: str | Path | None = None):
        return super().load(dataset_path or self.dataset_path)

    def make_loader(self):
        return D4rlLoader(self.dataset_path, device=self.device)
