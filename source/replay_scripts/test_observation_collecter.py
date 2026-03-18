from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from strategy import collect_observation_dataset


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
env = FullyObsWrapper(env)

result = collect_observation_dataset(
    env=env,
    output_path="tmps/one_room_s8_data.hdf5",
    max_episodes=3,
    max_episode_steps=20,
    seed=42,
    dataset_id="one-room-s8-data-v0",
)

print(
    "observation_collect_done "
    f"| episodes={result['episodes']} "
    f"| steps={result['steps']} "
    f"| dataset_id={result['dataset_id']} "
    f"| minari_path={result['minari_path']} "
    f"| path={result['path']}"
)
