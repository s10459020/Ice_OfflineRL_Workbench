from dataclasses import dataclass
from pathlib import Path

from ice_offline.dataset.loader.d4rl.loader import D4rlLoader
from ice_offline.config.paths import DATASETS_ROOT
from ice_offline.dataset._spec import Dataset


@dataclass
class HalfCheetahRandomDataset(Dataset):
    id: str = "halfcheetah_random"
    env_id: str = "HalfCheetah-v5"
    path: Path = DATASETS_ROOT / Path("d4rl/halfcheetah_random-v2/d4rl_data.hdf5")


    def make_loader(self):
        return D4rlLoader(self.path, device=self.device)




