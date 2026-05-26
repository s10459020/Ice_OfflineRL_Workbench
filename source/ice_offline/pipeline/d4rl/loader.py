from pathlib import Path

import h5py
import numpy as np


class D4rlLoader:
    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.buffer = self._build_buffer(self.dataset_path)

    def _build_buffer(self, dataset_path: Path) -> dict[str, np.ndarray]:
        with h5py.File(dataset_path, "r") as f:
            return {
                "observations": np.asarray(f["observations"]),
                "next_observations": np.asarray(f["next_observations"]),
                "actions": np.asarray(f["actions"]),
                "rewards": np.asarray(f["rewards"]),
                "terminations": np.asarray(f["terminals"]),
                "truncations": np.asarray(f["timeouts"]),
            }
