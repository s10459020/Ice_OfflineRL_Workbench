import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
from minigrid.wrappers import FullyObsWrapper

from visualization.minigrid import RenderDelayWrapper
from strategy import test


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human", max_steps=6000)
env = FullyObsWrapper(env)
env = RenderDelayWrapper(env, fps=2)

print("start | env=BabyAI-OneRoomS8-v0-fullobs | render_fps=2")
try:
    finished_episodes = test(
        env=env,
        max_episodes=30,
        max_episode_steps=200,
        seed=None,
        print_flag=True,
    )
finally:
    env.close()

print(f"finished_episodes={finished_episodes}")
