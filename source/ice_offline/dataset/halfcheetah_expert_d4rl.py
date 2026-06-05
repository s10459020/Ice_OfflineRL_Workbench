from dataclasses import dataclass
from pathlib import Path

from ice_offline.dataset.loader.d4rl.loader import D4rlLoader
from ice_offline.tools.paths import dataset_root
from ice_offline.dataset._spec import Dataset


@dataclass
class HalfCheetahExpertD4rlDataset(Dataset):
    id: str = "halfcheetah_expert_d4rl"
    env_id: str = "HalfCheetah-v5"
    path: Path = dataset_root() / Path("d4rl/halfcheetah_expert-v2/d4rl_data.hdf5")


    def make_loader(self):
        return D4rlLoader(self.path, device=self.device)


