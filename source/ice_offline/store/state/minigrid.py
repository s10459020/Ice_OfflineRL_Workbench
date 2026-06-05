from dataclasses import dataclass
from typing import Any

import numpy as np
from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj

from ice_offline.store.state._spec import State, StateIO

_ACTION_PICKUP = 3
_ACTION_DROP = 4
_OBJECT_EMPTY = OBJECT_TO_IDX["empty"]


@dataclass(frozen=True)
class MinigridState(State):
    mission: str
    agent_pos: tuple[int, int]
    agent_dir: int
    grid: np.ndarray
    carrying: tuple[int, int, int] | None

    def serialize(self) -> dict[str, Any]:
        return {
            "mission": self.mission.encode("utf-8"),
            "agent_pos": np.asarray(self.agent_pos, dtype=np.int16),
            "agent_dir": self.agent_dir,
            "grid": np.asarray(self.grid, dtype=np.int16),
            "carrying": np.asarray(self.carrying or (0, 0, 0), dtype=np.int16),
        }

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "MinigridState":
        encoded = tuple(payload["carrying"])
        carrying = None if encoded == (0, 0, 0) else encoded
        agent_pos_xy = tuple(payload["agent_pos"])
        mission_raw = payload["mission"]
        return cls(
            mission=mission_raw.decode("utf-8"),
            agent_pos=agent_pos_xy,
            agent_dir=payload["agent_dir"],
            grid=np.asarray(payload["grid"], dtype=np.int16),
            carrying=carrying,
        )


class MinigridStateIO(StateIO):
    def __init__(self, env: Any) -> None:
        self._env = env

    def get_state(self) -> MinigridState:
        base = self._env.unwrapped
        carrying = None if base.carrying is None else tuple(base.carrying.encode())
        return MinigridState(
            mission=base.mission,
            agent_pos=base.agent_pos,
            agent_dir=base.agent_dir,
            grid=np.asarray(base.grid.encode(), dtype=np.int8),
            carrying=carrying,
        )

    def set_state(self, state: MinigridState) -> None:
        base = self._env.unwrapped
        decoded = Grid.decode(np.asarray(state.grid, dtype=np.uint8))
        carrying = None
        if state.carrying is not None:
            carrying = WorldObj.decode(*state.carrying)

        base.grid = decoded[0]
        base.agent_pos = state.agent_pos
        base.agent_dir = state.agent_dir
        base.mission = state.mission
        base.carrying = carrying


class MinigridFullobsConverter:
    def convert_episode(self, trajectory: Any) -> list[MinigridState]:
        observations = trajectory.observations
        image_seq = observations["image"]
        dir_seq = observations["direction"]
        mission_seq = observations["mission"]
        act_seq = np.asarray(trajectory.actions, dtype=np.int64)
        num_states = len(dir_seq)

        grids: list[np.ndarray] = [
            np.asarray(image_seq[curr_index]).copy() for curr_index in range(num_states)
        ]
        agent_positions: list[tuple[int, int]] = [
            self._find_agent_pos(grids[curr_index]) for curr_index in range(num_states)
        ]
        carrying_seq = self._infer_carrying_sequence(
            grids, agent_positions, dir_seq, act_seq
        )

        states: list[MinigridState] = []
        for curr_index in range(num_states):
            states.append(
                MinigridState(
                    mission=mission_seq[curr_index],
                    agent_pos=agent_positions[curr_index],
                    agent_dir=dir_seq[curr_index],
                    grid=grids[curr_index],
                    carrying=carrying_seq[curr_index],
                )
            )
        return states

    def _infer_carrying_sequence(
        self,
        grids: list[np.ndarray],
        agent_positions: list[tuple[int, int]],
        dir_seq: Any,
        act_seq: np.ndarray,
    ) -> list[tuple[int, int, int] | None]:
        num_states = len(grids)
        carrying_seq: list[tuple[int, int, int] | None] = [None] * num_states
        carrying: tuple[int, int, int] | None = None

        for curr_index in range(1, num_states):
            prev_index = curr_index - 1
            action_prev = act_seq[prev_index]
            prev_front = self._front_cell(
                grids[prev_index], agent_positions[prev_index], dir_seq[prev_index]
            )
            curr_front = self._front_cell(
                grids[curr_index], agent_positions[curr_index], dir_seq[curr_index]
            )

            prev_empty = prev_front[0] == _OBJECT_EMPTY
            curr_empty = curr_front[0] == _OBJECT_EMPTY

            if action_prev == _ACTION_PICKUP and (not prev_empty) and curr_empty:
                carrying = tuple(prev_front.tolist())
            elif action_prev == _ACTION_DROP and prev_empty and (not curr_empty):
                carrying = None
            carrying_seq[curr_index] = carrying
        return carrying_seq

    def _front_cell(
        self, grid: np.ndarray, agent_pos: tuple[int, int], agent_dir: int
    ) -> np.ndarray:
        direction = DIR_TO_VEC[agent_dir]
        x = agent_pos[0] + direction[0]
        y = agent_pos[1] + direction[1]
        return grid[x, y]

    def _find_agent_pos(self, grid: np.ndarray) -> tuple[int, int]:
        agent_object_idx = OBJECT_TO_IDX["agent"]
        agent_coords = np.argwhere(grid[:, :, 0] == agent_object_idx)
        x, y = agent_coords[0]
        return x, y
