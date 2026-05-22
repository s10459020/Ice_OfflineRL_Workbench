import gymnasium as gym
import minigrid  # noqa: F401

import minari
from minigrid.wrappers import FullyObsWrapper

from ice_offline.env.common.mission_text_wrapper import MissionTextWrapper`nfrom ice_offline.env.common.no_jpeg_image_wrapper import NoJpegImageWrapper`nfrom ice_offline.tools.printer import print_stage


# ====================
# Environment Factory
# ====================
def make_env() -> gym.Env:
    env = FullyObsWrapper(gym.make("BabyAI-OneRoomS8-v0"))
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    return env


def make_eval_env() -> gym.Env:
    return gym.make("BabyAI-OneRoomS8-v0")


# ====================
# Rollout Helper
# ====================
def _rollout_episodes(
    collector: minari.DataCollector,
    seed: int,
    num_episodes: int,
    max_steps: int = 1024,
) -> tuple[int, int]:
    total_steps = 0

    for episode_idx in range(num_episodes):
        obs, _ = collector.reset(seed=seed + episode_idx)

        steps = 0
        for _ in range(max_steps):
            action = collector.action_space.sample()
            obs, _, terminated, truncated, _ = collector.step(action)

            _ = obs
            print(
                f"episode={episode_idx + 1}/{num_episodes} "
                f"step={steps + 1} action={action}"
            )

            steps += 1
            if terminated or truncated:
                break

        total_steps += steps

    return num_episodes, total_steps


def step_init_collector() -> minari.DataCollector:
    print_stage("Init Collector")
    return minari.DataCollector(make_env(), record_infos=False)


def step_reset_and_step(
    collector: minari.DataCollector,
    seed: int,
    num_episodes: int,
) -> None:
    print_stage("API: reset + step")
    episodes_0, total_steps_0 = _rollout_episodes(
        collector,
        seed=seed,
        num_episodes=num_episodes,
    )
    print(f"episodes={episodes_0} total_steps={total_steps_0}")


def step_create_dataset(
    collector: minari.DataCollector,
    dataset_id: str,
) -> minari.MinariDataset:
    print_stage("API: create_dataset")
    try:
        minari.delete_dataset(dataset_id)
    except Exception:
        pass
    dataset = collector.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random",
        author="local_test",
        author_email="local_test@example.com",
        code_permalink="https://example.com/oneroom-s8-fullobs-test",
        eval_env=make_eval_env(),
        description="DataCollector API smoke script",
    )
    print(f"created_dataset_id={dataset.spec.dataset_id}")
    print(f"created_total_episodes={dataset.total_episodes}")
    print(f"created_total_steps={dataset.total_steps}")
    return dataset


def step_add_to_dataset(
    dataset: minari.MinariDataset,
    seed: int,
    num_episodes: int,
) -> minari.DataCollector:
    print_stage("API: add_to_dataset")
    extra_collector = minari.DataCollector(make_env(), record_infos=False)
    episodes_1, total_steps_1 = _rollout_episodes(
        extra_collector,
        seed=seed,
        num_episodes=num_episodes,
    )
    print(f"extra_episodes={episodes_1} extra_total_steps={total_steps_1}")
    extra_collector.add_to_dataset(dataset)
    print(f"after_add_total_episodes={dataset.total_episodes}")
    print(f"after_add_total_steps={dataset.total_steps}")
    return extra_collector


def step_close(*collectors: minari.DataCollector) -> None:
    print_stage("API: close")
    for collector in collectors:
        collector.close()


def main() -> None:
    dataset_id = "minari_collector-v0"
    reset_seed = 55
    reset_episodes = 10
    add_seed = 99
    add_episodes = 10

    collector = step_init_collector()
    extra_collector = None
    try:
        step_reset_and_step(
            collector,
            seed=reset_seed,
            num_episodes=reset_episodes,
        )
        dataset = step_create_dataset(collector, dataset_id)
        extra_collector = step_add_to_dataset(
            dataset,
            seed=add_seed,
            num_episodes=add_episodes,
        )
    except Exception:
        try:
            minari.delete_dataset(dataset_id)
        except Exception:
            pass
        raise
    finally:
        if extra_collector is not None:
            step_close(extra_collector, collector)
        else:
            step_close(collector)
    print_stage("Done")


if __name__ == "__main__":
    main()

