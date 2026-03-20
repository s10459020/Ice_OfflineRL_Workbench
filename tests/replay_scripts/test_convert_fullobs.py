
import hashlib
import json
from pathlib import Path
import warnings

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from ice_offline.replay import StateDataset, convert_observation
from minigrid.wrappers import FullyObsWrapper
from ice_offline.tools import stage

from ice_offline.strategy import collect_dataset

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


env_id = "BabyAI-OneRoomS8-v0"
dataset_id = "one-room-s8-local-convert-test-v0"
episodes = 3

state_path = Path("tmps/one_room_s8_info.hdf5")
obs_path = Path("tmps/one_room_s8_data.hdf5")
converted_stem = f"{state_path.stem}_converted"
converted_state_path = state_path.with_name(f"{converted_stem}{state_path.suffix}")


###############################################################################
# STAGE: COLLECT
###############################################################################
stage("collect")
env = FullyObsWrapper(gym.make(env_id, render_mode="rgb_array"))
collect_steps = collect_dataset(
    env=env,
    state_output_path=state_path,
    observation_output_path=obs_path,
    max_episodes=episodes,
    seed=42,
)
print(
    "collect_done "
    f"| steps={collect_steps} "
    f"| info_path={state_path} "
    f"| obs_path={obs_path}"
)


###############################################################################
# STAGE: CONVERT
###############################################################################
stage("convert")
try:
    import minari
except ImportError as exc:  # pragma: no cover
    raise ImportError("minari is required for test_convert_fullobs.py.") from exc

local_dataset_id = dataset_id
dataset = minari.load_dataset(local_dataset_id)
total_to_convert = max(0, min(int(episodes), len(dataset)))
writer = StateDataset(converted_state_path, mode="w", flush_interval=1)
try:
    for episode_index in range(total_to_convert):
        states = convert_observation(dataset[episode_index].observations)
        writer.push_episode(states)
    writer.flush()
finally:
    writer.close()
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
stage("compare")
with StateDataset(state_path, mode="r") as original_state_manager, StateDataset(converted_state_path, mode="r") as converted_state_manager:
    if original_state_manager.num_episodes() != converted_state_manager.num_episodes():
        raise RuntimeError(
            f"episode count mismatch: original={original_state_manager.num_episodes()} converted={converted_state_manager.num_episodes()}"
        )

    all_equal = True
    for episode_index in range(original_state_manager.num_episodes()):
        original_states = list(original_state_manager.iter_episode_states(episode_index))
        converted_states = list(converted_state_manager.iter_episode_states(episode_index))
        if len(original_states) != len(converted_states):
            raise RuntimeError(
                f"episode length mismatch at episode={episode_index}: original={len(original_states)} converted={len(converted_states)}"
            )

        no_carry0 = original_state_manager.serialize_episode(
            episode_index,
        )
        no_carry1 = converted_state_manager.serialize_episode(
            episode_index,
        )
        equal = str(no_carry0) == str(no_carry1)
        all_equal = all_equal and equal

        mismatch = 0 if equal else len(original_states)
        print(
            f"episode={episode_index} "
            f"states={len(original_states)} "
            f"equal_no_carrying={equal} "
            f"mismatch_no_carrying={mismatch}"
        )

print(f"compare_done | all_equal_no_carrying={all_equal} | converted_info={converted_state_path}")
