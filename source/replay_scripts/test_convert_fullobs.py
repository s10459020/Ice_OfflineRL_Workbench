from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from replay import StateDatasetReader
from strategy import collect_dataset, convert_minari_fullobs_dataset

warnings.filterwarnings("ignore", message="Observation is not in observation space.*")
warnings.filterwarnings("ignore", message="Misconfigured dataset named .*")


def _banner(title: str) -> None:
    bar = "#" * 72
    print("\n" + bar)
    print(f"# {title.upper():^68} #")
    print(bar)


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


env_id = "BabyAI-OneRoomS8-v0"
dataset_id = "one-room-s8-local-convert-test-v0"
episodes = 3
max_episode_steps = 20

state_path = Path("tmps/one_room_s8_info.hdf5")
obs_path = Path("tmps/one_room_s8_data.hdf5")
converted_stem = f"{state_path.stem}_converted"
converted_state_path = state_path.with_name(f"{converted_stem}{state_path.suffix}")


_banner("collect")
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


_banner("convert")
local_dataset_id = str(obs_result.get("dataset_id", dataset_id))
convert_result = convert_minari_fullobs_dataset(
    dataset_id=local_dataset_id,
    output_path=converted_state_path,
    max_episodes=episodes,
)
print(
    "convert_fullobs_done "
    f"| dataset_id={convert_result['dataset_id']} "
    f"| converted_episodes={convert_result['converted_episodes']} "
    f"| path={convert_result['path']}"
)


_banner("compare")
with StateDatasetReader(state_path) as original_reader, StateDatasetReader(converted_state_path) as converted_reader:
    if original_reader.num_episodes != converted_reader.num_episodes:
        raise RuntimeError(
            f"episode count mismatch: original={original_reader.num_episodes} converted={converted_reader.num_episodes}"
        )

    all_equal = True
    for episode_index in range(original_reader.num_episodes):
        original_len = original_reader.episode_length(episode_index)
        converted_len = converted_reader.episode_length(episode_index)
        if original_len != converted_len:
            raise RuntimeError(
                f"episode length mismatch at episode={episode_index}: original={original_len} converted={converted_len}"
            )

        mismatch = 0
        carrying_gap = 0
        for state_index in range(original_len):
            s0 = original_reader.get_state(episode_index, state_index)
            s1 = converted_reader.get_state(episode_index, state_index)
            if _state_signature_no_carrying(s0) != _state_signature_no_carrying(s1):
                mismatch += 1
            if s0.carrying != s1.carrying:
                carrying_gap += 1

        equal = mismatch == 0
        all_equal = all_equal and equal
        print(
            f"episode={episode_index} "
            f"states={original_len} "
            f"equal_no_carrying={equal} "
            f"mismatch_no_carrying={mismatch} "
            f"carrying_diff={carrying_gap}"
        )

    print(f"compare_done | all_equal_no_carrying={all_equal} | converted_info={converted_state_path}")
