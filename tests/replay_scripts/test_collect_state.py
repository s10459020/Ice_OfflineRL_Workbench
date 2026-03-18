
import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.strategy import collect_dataset


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
env = FullyObsWrapper(env)

result = collect_dataset(
    env=env,
    collect_state=True,
    collect_observation=False,
    state_output_path="tmps/one_room_s8_info.hdf5",
    max_episodes=3,
    max_episode_steps=20,
    seed=42,
    flush_interval=10,
    print_flag=True,
)
env.close()

state_result = result["collect_state"] or {}
print(
    "state_collect_done "
    f"| episodes={result['episodes']} "
    f"| steps={result['steps']} "
    f"| path={state_result.get('path', '')}"
)
