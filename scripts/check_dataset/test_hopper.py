import gymnasium as gym


def main() -> None:
    env_id = "Hopper-v5"
    env = gym.make(env_id, render_mode="human")

    for _ in range(3):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main()
