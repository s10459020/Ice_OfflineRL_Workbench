from typing import Any

import gymnasium as gym
import minari

from ice_offline.env.common import insert_render_quiet_innermost
from .overlay_engine import OverlayEngine


# ------------------------------------------------------------------
# Unit Interface
# ------------------------------------------------------------------
class UnitLoaderInterface:
    # ====================
    # Loader Lifecycle
    # Order: on_env -> on_loader -> on_load -> on_seek -> on_render
    # ====================
    def on_env(self, base_env: gym.Env) -> None:
        pass
    
    def on_loader(self, engine: OverlayEngine, loader: "OverlayLoader") -> None:
        pass

    def on_load(self, datas: list[dict[str, Any]]) -> None:
        pass

    def on_seek(self, data: dict[str, Any]) -> None:
        pass

    def on_render(self, data: dict[str, Any]) -> None:
        pass


class OverlayLoader:
    """Offline dataset loader with overlay events: on_env/on_load/on_seek/on_render."""

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        dataset: str | Any,
        *,
        units: list[Any],
        render_mode: str = "rgb_array",
    ) -> None:
        for unit in units:
            if not isinstance(unit, UnitLoaderInterface):
                raise TypeError("each unit must implement UnitLoaderInterface")

        self.dataset = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset
        self.dataset_id: str = self.dataset.spec.dataset_id
        self.total_episodes: int = int(self.dataset.total_episodes)

        self.env = self.dataset.recover_environment(eval_env=True, render_mode=render_mode)
        self.env = insert_render_quiet_innermost(self.env)
        self._base_env = self.env.unwrapped
        self.engine = OverlayEngine(base_env=self._base_env, overlay_mode="frame")
        
        self._units = units
        self._list_snapshots: dict[str, Any] = {}
        for unit in self._units:
            unit.on_env(self._base_env)
            unit.on_loader(self.engine, self)

        self.data: dict[str, Any] = {}
        self._current_episode: int | None = None
        self._current_step: int = 0
        self._datas: list[dict[str, Any]] = []

        if self.total_episodes > 0:
            self.load(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_list(self, key: str, callback: Any) -> None:
        if key in self._list_snapshots:
            return
        self._list_snapshots[key] = callback

    def load(self, ep: int = 0) -> tuple[Any, dict[str, Any]]:
        trajectory = self.dataset[ep]
        count = len(trajectory.rewards)

        datas_dict: dict[str, list[Any]] = {
            "step_index": list(range(count + 1)),
            "observation": self._materialize_obs_seq(trajectory.observations, count),
            "info": self._materialize_info_seq(trajectory.infos, count),
            "action": [None] + list(trajectory.actions),
            "reward": [0.0] + list(trajectory.rewards),
            "terminated": [False] + list(trajectory.terminations),
            "truncated": [False] + list(trajectory.truncations),
        }
        datas_dict["done"] = [
            datas_dict["terminated"][i] or datas_dict["truncated"][i]
            for i in range(count + 1)
        ]
        for key, callback in self._list_snapshots.items():
            datas_dict[key] = callback(ep)

        self._datas = self._materialize_data_seq(datas_dict, count)

        self.env.reset()
        self._current_episode = ep

        for unit in self._units:
            unit.on_load(self._datas)
            
        self.seek(0)
        return self.data["observation"], self.data["info"]

    def seek(self, t: int) -> tuple[Any, Any, float, bool, bool, dict[str, Any]]:
        self._current_step = t
        self.data = self._datas[t]

        for unit in self._units:
            unit.on_seek(self.data)

        return (
            self.data["observation"],
            self.data["action"],
            self.data["reward"],
            self.data["terminated"],
            self.data["truncated"],
            self.data["info"],
        )

    def render(self) -> Any:
        for unit in self._units:
            unit.on_render(self.data)
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def get_episode_count(self) -> int:
        return self.total_episodes

    def get_current(self) -> dict[str, int | None]:
        return {"episode": self._current_episode, "transition": self._current_step}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _materialize_obs_seq(self, observations: Any, transition_count: int) -> list[Any]:
        if isinstance(observations, dict):
            return [{k: observations[k][i] for k in observations} for i in range(transition_count + 1)]
        return list(observations)

    def _materialize_info_seq(self, infos: Any, transition_count: int) -> list[dict[str, Any]]:
        if isinstance(infos, dict):
            return [{k: infos[k][i] for k in infos} for i in range(transition_count + 1)]
        return list(infos)

    def _materialize_data_seq(self, payload: dict[str, list[Any]], count: int) -> list[dict[str, Any]]:
        datas: list[dict[str, Any]] = []
        for i in range(count + 1):
            data: dict[str, Any] = {}
            for key, seq in payload.items():
                if i < len(seq):
                    data[key] = seq[i]
            datas.append(data)
        return datas
