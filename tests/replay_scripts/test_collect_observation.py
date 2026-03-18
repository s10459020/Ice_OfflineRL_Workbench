from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.strategy import collect_dataset


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
env = FullyObsWrapper(env)

result = collect_dataset(
    env=env,
    collect_observation=True,
    collect_state=False,
    observation_output_path="tmps/one_room_s8_data.hdf5",
    max_episodes=3,
    max_episode_steps=20,
    seed=42,
    dataset_id="one-room-s8-data-v0",
)
env.close()

obs_result = result["collect_observation"] or {}
print(
    "observation_collect_done "
    f"| episodes={result['episodes']} "
    f"| steps={result['steps']} "
    f"| dataset_id={obs_result.get('dataset_id', '')} "
    f"| minari_path={obs_result.get('minari_path', '')} "
    f"| path={obs_result.get('path', '')}"
)
