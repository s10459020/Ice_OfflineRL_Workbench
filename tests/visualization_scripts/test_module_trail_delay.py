import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.

from ice_offline.visualization.minigrid import RenderDelayWrapper, RenderOverlayWrapper, TrailWrapper
from ice_offline.strategy import test

env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human", max_steps=6000)
env = RenderOverlayWrapper(env)
env = TrailWrapper(env, clear_on_render=True, max_trails=64)
env = RenderDelayWrapper(env, fps=3, render_on_done=True)

print("start | env=BabyAI-OneRoomS8-v0 | trail=clear-on-render | max_trails=64 | delay_fps=3")
try:
    finished_episodes = test(
        env=env,
        max_episodes=200,
        max_episode_steps=5000,
        seed=None,
        print_flag=True,
    )
finally:
    env.close()

print(f"finished_episodes={finished_episodes}")
