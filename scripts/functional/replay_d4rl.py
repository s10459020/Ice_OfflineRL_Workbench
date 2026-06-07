import gymnasium as gym

from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.store.state.op_replayer import StateInjectWrapper
from ice_offline.store.state.hopper import HopperConverter, HopperState, HopperStateIO
from ice_offline.store.state.op_converter import StateConverter


def main() -> None:
    dataset = HopperMediumD4rlDataset()
    converter = StateConverter(dataset=dataset, converter_cls=HopperConverter)
    state_dataset = converter.convert()
    print(f"[convert] output={state_dataset.path}")
    print(f"[convert] all episodes done total={state_dataset.episode_count}")
    state_dataset.close()

    env = gym.make(dataset.env_id, render_mode="human")
    env = StateInjectWrapper(
        env=env,
        dataset=dataset,
        state_cls=HopperState,
        state_io_cls=HopperStateIO,
    )

    steps_total = 0
    try:
        for episode in range(dataset.episode_count):
            env.reset(options={"episode_index": episode})
            env.render()

            steps_episode = 0
            while True:
                _, reward, terminated, truncated, info = env.step(None)
                env.render()

                steps_total += 1
                steps_episode += 1
                print(
                    f"episode={episode} step={steps_episode} action={info.get('action')} "
                    f"reward={reward:.6f} terminated={terminated} truncated={truncated}"
                )

                if terminated or truncated:
                    break

            print(f"episode={episode} steps={steps_episode}")
    finally:
        env.close()

    print(f"episodes={dataset.episode_count} total_steps={steps_total}")


if __name__ == "__main__":
    main()
