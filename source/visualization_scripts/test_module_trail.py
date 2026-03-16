import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.

from visualization.minigrid import RenderOverlayWrapper, TrailWrapper
from strategy import test


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human", max_steps=6000)
env = RenderOverlayWrapper(env)
env = TrailWrapper(env, clear_on_render=False, max_trails=8)

print("start | env=BabyAI-OneRoomS8-v0 | trail=rolling(no-clear) | max_trails=8")
try:
    finished_episodes = test(
        env=env,
        max_episodes=20,
        max_episode_steps=20,
        seed=None,
        print_flag=True,
    )
finally:
    env.close()

print(f"finished_episodes={finished_episodes}")
