import gymnasium as gym


def main() -> None:
    env_id = "InvertedPendulum-v5"
    env = gym.make(env_id, render_mode="human")

    try:
        obs, _ = env.reset()
        print(f"env_id={env_id}")
        print(f"action = {env.action_space}")
        print(f"observation_type = {type(obs).__name__}")

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
