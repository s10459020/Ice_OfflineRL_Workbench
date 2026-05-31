from pathlib import Path

import h5py
import numpy as np
import torch

from ice_offline.dataset._spec import TorchBuffer


class D4rlLoader:
    def __init__(self, dataset_path: str | Path, device: str = "cpu") -> None:
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.buffer = self._load_buffer(self.dataset_path)
        self.env_id = ""
        self.obs_shape = tuple(int(x) for x in self.buffer.obs_list.shape[1:])
        self.act_shape = tuple(int(x) for x in self.buffer.act_list.shape[1:])
        self.obs_dim = int(np.prod(self.obs_shape)) if self.obs_shape else 1
        self.act_dim = int(np.prod(self.act_shape)) if self.act_shape else 1
        self.count = int(self.buffer.obs_list.shape[0])

    def _load_buffer(self, dataset_path: Path) -> TorchBuffer:
        with h5py.File(dataset_path, "r") as f:
            term = np.asarray(f["terminals"], dtype=np.bool_)
            trunc = np.asarray(f["timeouts"], dtype=np.bool_)
            return TorchBuffer(
                obs_list=torch.as_tensor(np.asarray(f["observations"]), dtype=torch.float32, device=self.device),
                next_obs_list=torch.as_tensor(np.asarray(f["next_observations"]), dtype=torch.float32, device=self.device),
                act_list=torch.as_tensor(np.asarray(f["actions"]), dtype=torch.float32, device=self.device),
                rew_list=torch.as_tensor(np.asarray(f["rewards"]), dtype=torch.float32, device=self.device),
                done_list=torch.as_tensor(np.logical_or(term, trunc), dtype=torch.float32, device=self.device),
            )
