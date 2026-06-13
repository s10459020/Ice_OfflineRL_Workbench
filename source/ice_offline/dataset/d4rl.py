from dataclasses import dataclass

from ice_offline.dataset.base import Dataset
from ice_offline.store.d4rl.loader import D4rlLoader


@dataclass
class D4rlDataset(Dataset):
    def make_loader(self):
        return D4rlLoader(self.path, device=self.device)
