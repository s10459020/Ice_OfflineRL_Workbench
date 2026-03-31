import gymnasium as gym
import minigrid  # noqa: F401
import minari

from ice_offline.env.common import MissionTextWrapper, NoJpegImageWrapper
from ice_offline.env.replay import StateRecordWrapper
from ice_offline.tools import print_stage


def main(
    dataset_id: str = "test_collect-v0",
    max_episodes: int = 10,
    seed: int = 123,
) -> None:
    # ---- Collect ----
    print_stage("Collect")
    steps = 0

    env = gym.make("BabyAI-OneRoomS8-v0")
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    env = StateRecordWrapper(env)
    collector = minari.DataCollector(env, record_infos=True)
    eval_env = gym.make("BabyAI-OneRoomS8-v0")

    try:
        for episode in range(1, max_episodes + 1):
            obs, _ = collector.reset(seed=seed + episode)
            done = False
            truncated = False
            while not (done or truncated):
                action = collector.action_space.sample()
                obs, _, done, truncated, _ = collector.step(action)
                _ = obs
                steps += 1

        try:
            minari.delete_dataset(dataset_id)
        except Exception:
            pass

        collector.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="random_policy",
            author="local_test",
            author_email="local_test@example.com",
            code_permalink="https://example.com/test_data_collect",
            eval_env=eval_env,
            description="collect smoke script without strategy dependency",
        )
    finally:
        eval_env.close()
        collector.close()

    print(f"collect_steps={steps}")

    # ---- Verify dataset ----
    print_stage("Verify Dataset")
    dataset = minari.load_dataset(dataset_id)
    print(f"dataset_id={dataset.spec.dataset_id}")
    print(f"total_episodes={dataset.total_episodes}")
    print(f"total_steps={dataset.total_steps}")


if __name__ == "__main__":
    main()
