import gymnasium as gym
import minigrid  # noqa: F401

from ice_offline.env.visualization import BasicUnit, OverlayWrapper


def run_overlay_wrapper(episodes: int = 3, max_steps: int = 100) -> None:
    env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
    env = OverlayWrapper(env, units=[BasicUnit()])

    try:
        for episode in range(episodes):
            _, _ = env.reset()
            print(f"\n=== episode {episode} start ===")

            done = False
            truncated = False
            steps = 0

            while not (done or truncated) and steps < max_steps:
                action = env.action_space.sample()
                _, reward, done, truncated, _ = env.step(action)
                env.render()

                print(
                    f"episode={episode} step={steps:03d} "
                    f"action={int(action)} reward={float(reward):.3f} "
                    f"done={done} truncated={truncated}"
                )
                steps += 1

            print(f"=== episode {episode} end: steps={steps} done={done} truncated={truncated} ===")
    finally:
        env.close()


if __name__ == "__main__":
    run_overlay_wrapper()
