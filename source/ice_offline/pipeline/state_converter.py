from pathlib import Path
from typing import Any

import h5py
import minari
import numpy as np
from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX

from ice_offline.data.state import State
from ice_offline.tools.paths import minari_root

_ACTION_PICKUP = 3
_ACTION_DROP = 4
_OBJECT_EMPTY = OBJECT_TO_IDX["empty"]


def convert_fullobs(dataset_source_id: str) -> Path:
    """Convert fullobs Minari trajectories into sidecar state_data.hdf5.

    The converter does not create a new Minari dataset. It only reads source
    episodes and writes state_data.hdf5 under source dataset folder.
    """
    source_dataset = minari.load_dataset(dataset_source_id)

    state_episodes: list[list[dict[str, Any]]] = []
    for trajectory in source_dataset.iterate_episodes():
        states = _convert_trajectory_to_states(trajectory)
        state_episodes.append([state.serialize() for state in states])

    return _save_state_data(dataset_source_id, state_episodes)


def _convert_trajectory_to_states(trajectory: Any) -> list[State]:
    observations = trajectory.observations
    image_seq = observations["image"]
    dir_seq = observations["direction"]
    mission_seq = observations["mission"]
    act_seq = np.asarray(trajectory.actions, dtype=np.int64)
    num_states = len(dir_seq)

    grids: list[np.ndarray] = [np.asarray(image_seq[curr_index]).copy() for curr_index in range(num_states)]
    agent_positions: list[tuple[int, int]] = [
        _find_agent_pos(grids[curr_index], curr_index=curr_index) for curr_index in range(num_states)
    ]
    carrying_seq = _infer_carrying_sequence(grids, agent_positions, dir_seq, act_seq)

    states: list[State] = []
    for curr_index in range(num_states):
        states.append(
            State(
                mission=mission_seq[curr_index],
                agent_pos=agent_positions[curr_index],
                agent_dir=dir_seq[curr_index],
                grid=grids[curr_index],
                carrying=carrying_seq[curr_index],
            )
        )
    return states


def _infer_carrying_sequence(
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
        prev_front = _front_cell(grids[prev_index], agent_positions[prev_index], dir_seq[prev_index])
        curr_front = _front_cell(grids[curr_index], agent_positions[curr_index], dir_seq[curr_index])

        prev_empty = prev_front[0] == _OBJECT_EMPTY
        curr_empty = curr_front[0] == _OBJECT_EMPTY

        if action_prev == _ACTION_PICKUP and (not prev_empty) and curr_empty:
            carrying = tuple(prev_front.tolist())
        elif action_prev == _ACTION_DROP and prev_empty and (not curr_empty):
            carrying = None

        carrying_seq[curr_index] = carrying
    return carrying_seq


def _front_cell(grid: np.ndarray, agent_pos: tuple[int, int], agent_dir: int) -> np.ndarray:
    direction = DIR_TO_VEC[agent_dir]
    x = agent_pos[0] + direction[0]
    y = agent_pos[1] + direction[1]
    return grid[x, y]


def _find_agent_pos(grid: np.ndarray, curr_index: int) -> tuple[int, int]:
    agent_object_idx = OBJECT_TO_IDX["agent"]
    agent_coords = np.argwhere(grid[:, :, 0] == agent_object_idx)
    x, y = agent_coords[0]
    return x, y


def _save_state_data(dataset_id: str, episodes: list[list[dict[str, Any]]]) -> Path:
    out_path = _resolve_state_path(dataset_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        for ep_idx, seq in enumerate(episodes):
            ep_group = f.require_group(f"episode_{ep_idx}")
            keys = seq[0].keys()
            for key in keys:
                values = [item[key] for item in seq]
                ep_group.create_dataset(key, data=np.asarray(values))
    return out_path


def _resolve_state_path(dataset_id: str) -> Path:
    base = minari_root()
    return base / dataset_id / "data" / "state_data.hdf5"
