import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from strategy import collect_state_dataset


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
env = FullyObsWrapper(env)

result = collect_state_dataset(
    env=env,
    output_path="tmps/state_dataset_demo.h5",
    max_episodes=1000,
    max_episode_steps=20,
    seed=42,
    write_interval=10,
    print_flag=True,
)
env.close()

print(
    "state_dataset_done "
    f"| episodes={result['episodes']} "
    f"| steps={result['steps']} "
    f"| path={result['path']}"
)
