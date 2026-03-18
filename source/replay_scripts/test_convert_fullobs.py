from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from replay import StateDatasetReader
from replay.state_types import AgentState
from minigrid.wrappers import FullyObsWrapper
from tools import print_banner

from strategy import (
    collect_dataset,
    convert_observation_tranjectory_to_state_tranjectory,
    serialize_state_tranjectory,
)

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None

warnings.filterwarnings("ignore", message="Observation is not in observation space.*")
warnings.filterwarnings("ignore", message="Misconfigured dataset named .*")


def _state_signature_no_carrying(state) -> str:
    grid = np.asarray(state.grid, dtype=np.uint8).copy()
    x, y = int(state.agent_pos[0]), int(state.agent_pos[1])
    if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
        # Normalize representation difference:
        # - state_capture grid does not write agent into grid
        # - fullobs image does write agent into grid
        grid[x, y, :] = np.array([1, 0, 0], dtype=np.uint8)
    payload = {
        "mission": str(state.mission),
        "agent_pos": [x, y],
        "agent_dir": int(state.agent_dir),
        "grid_shape": list(grid.shape),
        "grid_digest": hashlib.sha256(grid.tobytes()).hexdigest()[:16],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _write_episode(file, episode_index: int, states: list[AgentState]) -> None:
    group = file.create_group(f"episode_{episode_index}")
    group.attrs["num_states"] = len(states)
    utf8 = h5py.string_dtype(encoding="utf-8")

    missions = [str(state.mission) for state in states]
    positions = [(int(s.agent_pos[0]), int(s.agent_pos[1])) for s in states]
    directions = [int(s.agent_dir) for s in states]
    grids = [np.asarray(s.grid, dtype=np.uint8) for s in states]
    carrying = [json.dumps(s.carrying, ensure_ascii=True) for s in states]

    group.create_dataset("mission", data=np.asarray(missions, dtype=object), dtype=utf8)
    group.create_dataset("agent_pos", data=np.asarray(positions, dtype=np.int32))
    group.create_dataset("agent_dir", data=np.asarray(directions, dtype=np.int8))
    group.create_dataset("grid", data=np.asarray(grids, dtype=np.uint8), compression="gzip")
    group.create_dataset("carrying", data=np.asarray(carrying, dtype=object), dtype=utf8)


env_id = "BabyAI-OneRoomS8-v0"
dataset_id = "one-room-s8-local-convert-test-v0"
episodes = 3
max_episode_steps = 20

state_path = Path("tmps/one_room_s8_info.hdf5")
obs_path = Path("tmps/one_room_s8_data.hdf5")
converted_stem = f"{state_path.stem}_converted"
converted_state_path = state_path.with_name(f"{converted_stem}{state_path.suffix}")


###############################################################################
# STAGE: COLLECT
###############################################################################
print_banner("collect")
env = FullyObsWrapper(gym.make(env_id, render_mode="rgb_array"))
collect_result = collect_dataset(
    env=env,
    collect_state=True,
    collect_observation=True,
    state_output_path=state_path,
    observation_output_path=obs_path,
    dataset_id=dataset_id,
    max_episodes=episodes,
    max_episode_steps=max_episode_steps,
    seed=42,
    print_flag=False,
)
obs_result = collect_result["collect_observation"] or {}
print(
    "collect_done "
    f"| episodes={collect_result['episodes']} "
    f"| steps={collect_result['steps']} "
    f"| info_path={collect_result['collect_state']['path']} "
    f"| obs_path={obs_result.get('path', '')} "
    f"| local_minari_path={obs_result.get('minari_path', '')}"
)


###############################################################################
# STAGE: CONVERT
###############################################################################
print_banner("convert")
if h5py is None:  # pragma: no cover
    raise ImportError("h5py is required for test_convert_fullobs.py.") from _H5PY_IMPORT_ERROR
try:
    import minari
except ImportError as exc:  # pragma: no cover
    raise ImportError("minari is required for test_convert_fullobs.py.") from exc

local_dataset_id = str(obs_result.get("dataset_id", dataset_id))
dataset = minari.load_dataset(local_dataset_id)
total_to_convert = max(0, min(int(episodes), len(dataset)))
converted_state_path.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(converted_state_path, "w") as file:
    file.attrs["format"] = "state_dataset_v1"
    file.attrs["total_episodes"] = int(total_to_convert)
    for episode_index in range(total_to_convert):
        states = convert_observation_tranjectory_to_state_tranjectory(dataset[episode_index].observations)
        _write_episode(file, episode_index=episode_index, states=states)
convert_result = {"dataset_id": local_dataset_id, "converted_episodes": total_to_convert, "path": str(converted_state_path)}
print(
    "convert_fullobs_done "
    f"| dataset_id={convert_result['dataset_id']} "
    f"| converted_episodes={convert_result['converted_episodes']} "
    f"| path={convert_result['path']}"
)


###############################################################################
# STAGE: COMPARE
###############################################################################
print_banner("compare")
with StateDatasetReader(state_path) as original_reader, StateDatasetReader(converted_state_path) as converted_reader:
    if original_reader.num_episodes != converted_reader.num_episodes:
        raise RuntimeError(
            f"episode count mismatch: original={original_reader.num_episodes} converted={converted_reader.num_episodes}"
        )

    all_equal = True
    for episode_index in range(original_reader.num_episodes):
        original_states = list(original_reader.iter_episode_states(episode_index))
        converted_states = list(converted_reader.iter_episode_states(episode_index))
        if len(original_states) != len(converted_states):
            raise RuntimeError(
                f"episode length mismatch at episode={episode_index}: original={len(original_states)} converted={len(converted_states)}"
            )

        no_carry0 = serialize_state_tranjectory(
            original_states,
            include_payload=False,
            include_signature=True,
            ignore_carrying=True,
            normalize_agent_cell=True,
        )
        no_carry1 = serialize_state_tranjectory(
            converted_states,
            include_payload=False,
            include_signature=True,
            ignore_carrying=True,
            normalize_agent_cell=True,
        )
        equal = str(no_carry0.get("signature", "")) == str(no_carry1.get("signature", ""))
        all_equal = all_equal and equal

        full0 = serialize_state_tranjectory(original_states, include_payload=True, include_signature=False)
        full1 = serialize_state_tranjectory(converted_states, include_payload=True, include_signature=False)
        carrying_gap = 0
        payload0 = full0.get("payload", [])
        payload1 = full1.get("payload", [])
        for i in range(min(len(payload0), len(payload1))):
            c0 = payload0[i].get("carrying") if isinstance(payload0[i], dict) else None
            c1 = payload1[i].get("carrying") if isinstance(payload1[i], dict) else None
            if c0 != c1:
                carrying_gap += 1

        mismatch = 0 if equal else len(original_states)
        print(
            f"episode={episode_index} "
            f"states={len(original_states)} "
            f"equal_no_carrying={equal} "
            f"mismatch_no_carrying={mismatch} "
            f"carrying_diff={carrying_gap}"
        )

print(f"compare_done | all_equal_no_carrying={all_equal} | converted_info={converted_state_path}")
