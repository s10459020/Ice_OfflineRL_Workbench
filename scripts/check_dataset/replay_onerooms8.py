import gymnasium as gym
import minigrid  # noqa: F401

from ice_offline.env.common.render_quiet_wrapper import insert_render_quiet_innermost
from ice_offline.pipeline.state.oneroom_s8 import OneroomS8State, OneroomS8StateIO
from ice_offline.pipeline.state_converter import convert_fullobs
from ice_offline.pipeline.state_op.state_injector import StateInjectWrapper


DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
ENV_ID = "BabyAI-OneRoomS8-v0"


def main() -> None:
    state_path = convert_fullobs(dataset_source_id=DATASET_ID)
    print(f"state_data_path={state_path}")

    env = gym.make(ENV_ID, render_mode="human")
    env = insert_render_quiet_innermost(env)
    env = StateInjectWrapper(
        env=env,
        dataset_id=DATASET_ID,
        state_cls=OneroomS8State,
        state_io_cls=OneroomS8StateIO,
    )

    steps_total = 0
    try:
        for episode in range(env.total_episodes):
            env.reset(options={"episode_index": episode})
            env.render()

            steps_episode = 0
            while True:
                _, reward, terminated, truncated, info = env.step(None)
                steps_total += 1
                steps_episode += 1
                env.render()
                print(
                    f"episode={episode} step={steps_episode} action={info.get('action')} "
                    f"reward={reward:.3f} terminated={terminated} truncated={truncated}"
                )
                if terminated or truncated:
                    break
            print(f"episode={episode} steps={steps_episode}")
    finally:
        env.close()

    print(f"total_steps={steps_total}")


if __name__ == "__main__":
    main()
