import gymnasium as gym
import minari


ENV_ID = "Hopper-v5"
DATASET_ID = "mujoco/hopper/medium-v0"


def online() -> None:
    env = gym.make(ENV_ID, render_mode="human")
    try:
        for _ in range(3):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    finally:
        env.close()


def offline(dataset_id: str = DATASET_ID) -> None:
    dataset = minari.load_dataset(dataset_id, download=True)
    env = dataset.recover_environment()
    try:
        for _ in dataset.iterate_episodes():
            break
        env.reset()
    finally:
        env.close()


def main() -> None:
    offline()


if __name__ == "__main__":
    main()
