import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

def minigrid_state_line(obs: dict) -> str:
    return (
        f"img_shape={obs['image'].shape}, "
        f"dir={int(obs['direction'])}, "
        f"mission={obs['mission']}"
    )

env = gym.make("BabyAI-OneRoomS8-v0")
env = FullyObsWrapper(env)

state, _ = env.reset()
for step in range(1, 11):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    print(
        f"step={step} | s={minigrid_state_line(state)} | a={action} | "
        f"r={float(reward):.3f} | s'={minigrid_state_line(next_state)}"
    )
    state = next_state
    if terminated or truncated:
        state, _ = env.reset()

env.close()
