from dataclasses import dataclass
from pathlib import Path

from ice_offline.dataset.loader.d4rl.loader import D4rlLoader
from ice_offline.config.paths import DATASETS_ROOT
from ice_offline.dataset._spec import Dataset


@dataclass
class Walker2dExpertD4rlDataset(Dataset):
    id: str = "walker2d_expert_d4rl"
    env_id: str = "Walker2d-v5"
    path: Path = DATASETS_ROOT / Path("d4rl/walker2d_expert-v2/d4rl_data.hdf5")


    def make_loader(self):
        return D4rlLoader(self.path, device=self.device)




