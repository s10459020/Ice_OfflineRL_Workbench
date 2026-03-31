import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
from minigrid.wrappers import FullyObsWrapper

from ice_offline.visualization.minigrid import RenderDelayWrapper
from ice_offline.strategy import test


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human", max_steps=6000)
env = FullyObsWrapper(env)
env = RenderDelayWrapper(env, fps=2)

print("start | env=BabyAI-OneRoomS8-v0-fullobs | render_fps=2")
try:
    finished_steps = test(
        env=env,
        max_episodes=30,
        seed=None,
        print_interval=1,
    )
finally:
    env.close()

print(f"finished_steps={finished_steps}")
