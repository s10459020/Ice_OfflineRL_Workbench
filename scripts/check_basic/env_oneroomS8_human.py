import gymnasium as gym
import minigrid  # noqa: F401


def main() -> None:
    env_id = "BabyAI-OneRoomS8-v0"
    env = gym.make(env_id, render_mode="human")

    try:
        obs, _ = env.reset()
        print(f"env_id={env_id}")

        for step in range(1, 2000):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            print(f"step={step} a={action} r={reward:.3f} done={done}")
            if done:
                print("episode_end -> reset")
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
