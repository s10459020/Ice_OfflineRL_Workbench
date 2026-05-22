import gymnasium as gym

def main() -> None:
    env_id = "Walker2d-v5"
    env = gym.make(env_id, render_mode="human")

    try:
        obs, _ = env.reset()
        print("env_id={env_id}")
        print(f"action = {env.action_space}")
        print(f"observation_type = {type(obs).__name__}")

        for step in range(1, 2000):
            action = env.action_space.sample()
            _, reward, term, trun, _ = env.step(action)
            done = term or trun
            print(f"step={step} a={action} r={reward:3f}, done={done}")
            if done:
                print("episode_end -> reset")
                env.reset()
    finally:
        env.cloes()


if __name__ == "__main__":
    main()
            