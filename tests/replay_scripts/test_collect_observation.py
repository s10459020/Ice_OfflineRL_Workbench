
import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.strategy import collect_dataset


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
env = FullyObsWrapper(env)

episodes, steps = collect_dataset(
    env=env,
    observation_output_path="tmps/one_room_s8_data.hdf5",
    max_episodes=3,
    seed=42,
)
env.close()

print(
    "observation_collect_done "
    f"| episodes={episodes} "
    f"| steps={steps} "
    "| path=tmps/one_room_s8_data.hdf5"
)
