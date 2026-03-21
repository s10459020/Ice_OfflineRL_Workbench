import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
from minigrid.wrappers import FullyObsWrapper


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)

try:
    obs, _ = env.reset()
    print("env=BabyAI-OneRoomS8-v0-fullobs")
    print(f"action_space={env.action_space}")
    print(f"observation_type={type(obs).__name__}")
    print(f"observation_keys={list(obs.keys())}")
    print(f"obs.image shape={obs['image'].shape} dtype={obs['image'].dtype}")
    print(f"obs.direction={obs['direction']} ({type(obs['direction']).__name__})")
    print(f"obs.mission={obs['mission']}")
    print("obs.image[:, :, 0] (object index map):")
    for row in obs["image"][:, :, 0]:
        print(" ".join(f"{int(v):2d}" for v in row))

    episodes = 0
    for _ in range(10):
        env.reset()
        episodes += 1
        for _ in range(10):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    print(f"episodes={episodes}")
finally:
    env.close()
