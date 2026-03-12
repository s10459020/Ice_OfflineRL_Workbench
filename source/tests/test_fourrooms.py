import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.


def run_random_policy(steps: int = 100, render_mode: str = "human") -> None:
    env = gym.make("MiniGrid-FourRooms-v0", render_mode=render_mode)
    obs, _ = env.reset()
    print(f"reset | obs_keys={list(obs.keys())}")

    for step in range(1, steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(
            f"step={step} action={action} reward={float(reward):.3f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated:
            obs, _ = env.reset()
            print(f"reset | obs_keys={list(obs.keys())}")

    env.close()


if __name__ == "__main__":
    run_random_policy(steps=100, render_mode="human")
