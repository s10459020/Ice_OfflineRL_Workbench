import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper


def ndarray_text(array: np.ndarray) -> str:
    if array.ndim == 3:
        channels = array.shape[2]
        sections = []
        for channel_idx in range(channels):
            channel = array[:, :, channel_idx]
            lines = [" ".join(f"{int(v):2d}" for v in row) for row in channel]
            sections.append(f"channel={channel_idx}\n" + "\n".join(lines))
        return "\n\n".join(sections)
    if array.ndim == 2:
        return "\n".join(" ".join(f"{int(v):2d}" for v in row) for row in array)
    return str(array)


def observation_text(obs: dict) -> str:
    sections = []
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            sections.append(
                f"observation.{key} = shape={value.shape}, dtype={value.dtype}\n{ndarray_text(value)}"
            )
        else:
            sections.append(
                f"observation.{key} = ({type(value).__name__}) {value}"
            )
    return "\n\n".join(sections)


def main() -> None:
    env_id = "BabyAI-OneRoomS8-v0"
    env = gym.make(env_id)
    env = FullyObsWrapper(env)

    try:
        obs, _ = env.reset()

        print(f"env_id={env_id}")
        print("wrapper=FullyObsWrapper")
        print("---------action-------")
        print(f"action = {env.action_space}")
        print("---------observation-----------")
        print(f"observation = {env.observation_space}")
        print(f"max_steps={env.unwrapped.max_steps}")
        print("---------observation value-----------")
        print(observation_text(obs))
    finally:
        env.close()


if __name__ == "__main__":
    main()
