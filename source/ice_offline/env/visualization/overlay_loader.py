from typing import Any

import gymnasium as gym
import minari

from ice_offline.env.common import StateIOWrapper, insert_render_quiet_innermost
from ice_offline.env.model import State
from .overlay_engine import OverlayEngine, UnitRegisterInterface


# ------------------------------------------------------------------
# Unit Interface
# ------------------------------------------------------------------
class UnitLoaderInterface:
    def on_env(self, base_env: gym.Env) -> None:
        pass

    def on_load(
        self,
        states: list[State],
        actions: list[Any],
        rewards: list[float],
        dones: list[bool],
        infos: list[dict[str, Any]],
    ) -> None:
        pass

    def on_seek(self, transition_index: int) -> None:
        pass

    def on_render(self, state: State, info: dict[str, Any]) -> None:
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
            if not isinstance(unit, UnitLoaderInterface) or not isinstance(unit, UnitRegisterInterface):
                raise TypeError("each unit must implement UnitLoaderInterface and UnitRegisterInterface")

        self.dataset = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset
        self.total_episodes: int = int(self.dataset.total_episodes)

        env = self.dataset.recover_environment(eval_env=True, render_mode=render_mode)
        env = insert_render_quiet_innermost(env)
        self._state_io = StateIOWrapper(env)
        self.env = self._state_io
        self._base_env = self.env.unwrapped

        self.engine = OverlayEngine(base_env=self._base_env, overlay_mode="frame")
        self._units = units
        for unit in self._units:
            unit.on_env(self._base_env)
            unit.register_engine(self.engine)

        self._current_episode: int | None = None
        self._current_step: int = 0
        self._transition_count: int = 0
        self._states: list[State] = []
        self._observations: list[Any] = []
        self._infos: list[dict[str, Any]] = []
        self._actions: list[Any] = []
        self._rewards: list[float] = []
        self._terminations: list[bool] = []
        self._truncations: list[bool] = []

        if self.total_episodes > 0:
            self.load(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, ep: int = 0) -> tuple[Any, dict[str, Any]]:
        trajectory = self.dataset[int(ep)]
        self._actions = list(trajectory.actions)
        self._rewards = list(trajectory.rewards)
        self._transition_count = len(self._rewards)
        self._terminations = list(trajectory.terminations)
        self._truncations = list(trajectory.truncations)
        self._observations = self._materialize_obs_seq(trajectory.observations, self._transition_count)
        self._infos = self._materialize_info_seq(trajectory.infos, self._transition_count)
        self._states = self._states_from_info(trajectory.infos, self._transition_count)

        # Ensure wrapper order checks (e.g., OrderEnforcer) are satisfied before rendering.
        self.env.reset()
        self._state_io.set_state(self._states[0])
        self._current_episode = int(ep)
        self._current_step = 0

        dones = [bool(t or u) for t, u in zip(self._terminations, self._truncations)]
        for unit in self._units:
            unit.on_load(self._states, self._actions, self._rewards, dones, self._infos)
        return self._observations[0], self._infos[0]

    def seek(self, t: int) -> tuple[Any, Any, float, bool, bool, dict[str, Any]] | None:
        if t == 0:
            self._state_io.set_state(self._states[0])
            self._current_step = 0
            for unit in self._units:
                unit.on_seek(0)
            return None

        obs = self._observations[t]
        reward = float(self._rewards[t - 1])
        terminated = bool(self._terminations[t - 1])
        truncated = bool(self._truncations[t - 1])
        info = dict(self._infos[t])
        action = self._actions[t - 1]

        self._state_io.set_state(self._states[t])
        self._current_step = t
        for unit in self._units:
            unit.on_seek(int(t))
        return (obs, action, reward, terminated, truncated, info)

    def render(self) -> Any:
        state = self._state_io.get_state()
        for unit in self._units:
            unit.on_render(state, {})
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
    def _states_from_info(self, infos: Any, transition_count: int) -> list[State]:
        state_payload = infos["state"]
        payload_seq = [{k: state_payload[k][i] for k in state_payload} for i in range(transition_count + 1)]
        return [State.from_serialized(payload) for payload in payload_seq]

    def _materialize_obs_seq(self, observations: Any, transition_count: int) -> list[Any]:
        if isinstance(observations, dict):
            return [{k: observations[k][i] for k in observations} for i in range(transition_count + 1)]
        return list(observations)

    def _materialize_info_seq(self, infos: Any, transition_count: int) -> list[dict[str, Any]]:
        return [self._index_payload(infos, i) for i in range(transition_count + 1)]

    def _index_payload(self, payload: Any, index: int) -> Any:
        if isinstance(payload, dict):
            return {k: self._index_payload(v, index) for k, v in payload.items()}
        return payload[index]
