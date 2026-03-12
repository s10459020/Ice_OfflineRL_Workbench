import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.

from visualizers.minigrid import RenderDelayWrapper


def run_with_delay(steps: int = 60, fps: int = 3) -> None:
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="human", max_steps=1000)
    env = RenderDelayWrapper(env, fps=fps)

    obs, _ = env.reset()
    print(f"reset | obs_keys={list(obs.keys())} | render_fps={fps}")

    env.render()
    for step in range(1, steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        print(
            f"step={step} action={action} reward={float(reward):.3f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated:
            obs, _ = env.reset()
            env.render()
            print(f"reset | obs_keys={list(obs.keys())}")

    env.close()


if __name__ == "__main__":
    run_with_delay(steps=6000, fps=5)
