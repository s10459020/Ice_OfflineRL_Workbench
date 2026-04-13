import gymnasium as gym
import numpy as np


def ndarray_text(array: np.ndarray) -> str:
    if array.ndim == 1:
        return " ".join(f"{float(v): .6f}" for v in array)
    return str(array)


def main() -> None:
    env_id = "InvertedPendulum-v5"
    env = gym.make(env_id)

    try:
        obs, _ = env.reset()

        print(f"env_id={env_id}")
        print("---------action-------")
        print(f"action = {env.action_space}")
        print("---------observation-----------")
        print(f"observation = {env.observation_space}")
        print("---------reset observation-----------")
        print(f"shape={obs.shape}, dtype={obs.dtype}")
        print(ndarray_text(obs))
    finally:
        env.close()


if __name__ == "__main__":
    main()
