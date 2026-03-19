
import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.strategy import collect_dataset


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
env = FullyObsWrapper(env)

episodes, steps = collect_dataset(
    env=env,
    state_output_path="tmps/one_room_s8_info.hdf5",
    max_episodes=3,
    seed=42,
    flush_interval=10,
    print_interval=1,
)
env.close()

print(
    "state_collect_done "
    f"| episodes={episodes} "
    f"| steps={steps} "
    "| path=tmps/one_room_s8_info.hdf5"
)
