from ice_offline.pipeline.state.hopper import HopperState, HopperStateIO
from ice_offline.pipeline.state.op_replayer import make_replayer


#DATASET_ID = "mujoco/hopper/medium-v0"
DATASET_ID = "test/hopper_simple_random-v0"
STATE_CLS = HopperState
STATE_IO_CLS = HopperStateIO

def replay(dataset_id: str, state_cls: type[HopperState], state_io_cls: type[HopperStateIO]) -> None:
    env = make_replayer(
        dataset_id=dataset_id,
        state_cls=state_cls,
        state_io_cls=state_io_cls,
    )

    steps_total = 0
    try:
        for episode in range(env.total_episodes):
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

    print(f"dataset_id={dataset_id} episodes={env.total_episodes} total_steps={steps_total}")


if __name__ == "__main__":
    replay(
        DATASET_ID,
        state_cls=STATE_CLS,
        state_io_cls=STATE_IO_CLS,
    )
