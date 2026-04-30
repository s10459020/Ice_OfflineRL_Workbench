import gymnasium as gym
import minigrid  # noqa: F401


def main() -> None:
    env = gym.make("BabyAI-OneRoomS8-v0")

    env.reset()
    for step in range(1, 6):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"step={step} a={action} r={reward:.3f} done={done}")
        if done:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
