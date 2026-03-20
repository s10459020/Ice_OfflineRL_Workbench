from pathlib import Path
from typing import Any

import gymnasium as gym

from ice_offline.replay.state_dataset import StateDataset
from ice_offline.replay.state_capture_wrapper import ensure_state_capture


class StateCollector(gym.Wrapper):
    """Collect state trajectories from info["state"] during reset/step."""

    def __init__(
        self,
        env: gym.Env,
        output_path: str | Path,
        *,
        flush_interval: int = 0,
        compression: str | None = "gzip",
    ) -> None:
        super().__init__(ensure_state_capture(env))
        self.output_path = Path(output_path)
        self.state_dataset = StateDataset(
            self.output_path,
            mode="w",
            flush_interval=flush_interval,
            compression=compression,
        )
        self._episode_active = False
        self._closed = False

    @property
    def episodes(self) -> list[list[Any]]:
        return self.state_dataset.episodes

    def reset(self, **kwargs: Any):
        if self._episode_active:
            self.state_dataset.end_episode()
            self._episode_active = False

        obs, info = self.env.reset(**kwargs)
        self.state_dataset.push_state(info["state"])
        self._episode_active = True
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state_dataset.push_state(info["state"])
        if terminated or truncated:
            self.state_dataset.end_episode()
            self._episode_active = False
        return obs, reward, terminated, truncated, info

    def flush(self) -> None:
        self.state_dataset.flush()

    def close_writer(self) -> None:
        if self._closed:
            return
        self.flush()
        self.state_dataset.close()
        self._closed = True

    def close(self) -> None:
        self.close_writer()
        super().close()
