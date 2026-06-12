import json
import shutil

import gymnasium as gym
import tempfile
from pathlib import Path

import h5py
from ice_offline.config.paths import DATASETS_ROOT
import numpy as np


class EvalCollector(gym.Wrapper):
    def __init__(self, env: gym.Env, temp_dir: Path = DATASETS_ROOT, resume_path: Path | None = None):
        super().__init__(env)
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = tempfile.TemporaryDirectory(dir=temp_dir.parent)
        self._temp_path = Path(self._temp_dir.name) / "main_data.hdf5"
        if resume_path is not None and resume_path.exists():
            shutil.copy2(resume_path, self._temp_path)

        self._episodes: list[dict] = []
        self._current: dict | None = None


    # ====================
    # Override
    # ====================
    def reset(self, *args, **kwargs):
        o, info = self.env.reset(*args, **kwargs)    
        self._new_episode(o)
        return o, info
    
    def step(self, a, *args, **kwargs):
        o, r, term, trun, info = self.env.step(a, *args, **kwargs)
        self._add_episode(o, a, r, term, trun)
        return o, r, term, trun, info
    

    # ====================
    # Public API
    # ====================
    def flush(self, episode: int) -> None:
        self._end_episode()
        with h5py.File(self._temp_path, "a", track_order=True) as f:
            for i, data in enumerate(self._episodes):
                key = f"episode_{episode}_{i}"
                group = f.create_group(key)

                for name, values in data.items():
                    group.create_dataset(name, data = np.asarray(values))

        self._episodes.clear()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._temp_path, path)
        
        metadata = {
            "id": "not imprement id",
            "env_id": self.env.spec.id,
            "agent_id": "not imprement agent id",
        }
        metadata_path = path.parent / "metadata.json"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, ensure_ascii=False)
            file.write("\n")

        self._temp_dir.cleanup()
    
    
    # ====================
    # Private Methods
    # ====================
    def _new_episode(self, o):
        self._end_episode()
        self._current = {
            "observations": [o],
            "actions": [],
            "rewards": [],
            "truncations": [],
            "terminations": [],
            "infos": [],
        }

    def _add_episode(self, on, a, r, term, trun):
        self._current["observations"].append(on)
        self._current["actions"].append(a)
        self._current["rewards"].append(r)
        self._current["terminations"].append(term)
        self._current["truncations"].append(trun)
        
    def _end_episode(self):
        if self._current is not None:
            self._episodes.append(self._current)
            self._current = None
