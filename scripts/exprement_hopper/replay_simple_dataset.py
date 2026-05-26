import minari
import gymnasium as gym

from ice_offline.pipeline.state.hopper import HopperState, HopperStateIO
from ice_offline.pipeline.state.hopper import HopperConverter
from ice_offline.pipeline.state.op_converter import StateConverter
from ice_offline.pipeline.state.op_injector import StateInjectWrapper


DATASET_ID = "mujoco/hopper/simple-v0"
ENV_ID = "Hopper-v5"


if __name__ == "__main__":
    dataset = minari.load_dataset(DATASET_ID, download=True)
    converter = StateConverter(dataset=dataset, converter_cls=HopperConverter)
    episodes = dataset.total_episodes

    state_dataset = converter.convert()
    print(f"[convert] output={state_dataset.path}")
    print(f"[convert] all episodes done total={episodes}")

    env = gym.make(ENV_ID, render_mode="human")
    env = StateInjectWrapper(
        env=env,
        dataset_id=DATASET_ID,
        state_cls=HopperState,
        state_io_cls=HopperStateIO,
    )

    steps_total = 0
    try:
        for episode in range(episodes):
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

    print(f"episodes={episodes} total_steps={steps_total}")



