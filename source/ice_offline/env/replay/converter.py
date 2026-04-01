from typing import Any

import minari
import numpy as np
from gymnasium import spaces
from minari.data_collector.episode_buffer import EpisodeBuffer
from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX

from ice_offline.env.model import State

_ACTION_PICKUP = 3
_ACTION_DROP = 4
_OBJECT_EMPTY = OBJECT_TO_IDX["empty"]


def convert_fullobs(dataset_source_id: str, dataset_target_id: str) -> Any:
    source_dataset = minari.load_dataset(dataset_source_id)

    episode_buffers: list[EpisodeBuffer] = []
    for trajectory in source_dataset.iterate_episodes():
        states = _convert_trajectory_to_states(trajectory)
        buffer = EpisodeBuffer(
            observations=to_nojpeg_observations(trajectory.observations),
            actions=trajectory.actions,
            rewards=list(trajectory.rewards),
            terminations=list(trajectory.terminations),
            truncations=list(trajectory.truncations),
            infos=_merge_infos_with_state(trajectory, states),
        )
        episode_buffers.append(buffer)

    try:
        minari.delete_dataset(dataset_target_id)
    except Exception:
        pass

    target_dataset = minari.create_dataset_from_buffers(
        dataset_id=dataset_target_id,
        env=source_dataset.spec.env_spec,
        eval_env=source_dataset.spec.env_spec,
        buffer=episode_buffers,
        algorithm_name="fullobs_to_state",
        author="ice_offline",
        author_email="local_test@example.com",
        code_permalink="https://example.com/ice-offline-converter",
        description=f"Converted from {dataset_source_id}",
        observation_space=to_nojpeg_observation_space(source_dataset.observation_space),
        action_space=source_dataset.action_space,
    )
    return target_dataset


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


def _merge_infos_with_state(trajectory: Any, states: list[State]) -> dict[str, Any]:
    infos = trajectory.infos or {}
    serialized_states = [state.serialize() for state in states]
    infos["state"] = {
        "mission": [state["mission"] for state in serialized_states],
        "agent_pos": [state["agent_pos"] for state in serialized_states],
        "agent_dir": [state["agent_dir"] for state in serialized_states],
        "grid": [state["grid"] for state in serialized_states],
        "carrying": [state["carrying"] for state in serialized_states],
    }
    return infos


def to_nojpeg_observations(observations: Any) -> Any:
    converted = dict(observations)
    converted["image"] = converted["image"].astype(np.int16, copy=False)
    return converted


def to_nojpeg_observation_space(observation_space: Any) -> Any:
    spaces_dict = dict(observation_space.spaces)
    image_space = spaces_dict["image"]
    image_low = image_space.low.min()
    image_high = image_space.high.max()
    spaces_dict["image"] = spaces.Box(
        low=image_low,
        high=image_high,
        shape=image_space.shape,
        dtype=np.int16,
    )
    return spaces.Dict(spaces_dict)
