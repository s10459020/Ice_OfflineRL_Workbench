import gymnasium as gym
import minigrid  # noqa: F401
import minari
from minigrid.wrappers import FullyObsWrapper

from ice_offline.tools import MissionTextWrapper, NoJpegImageWrapper


# ====================
# Console Helpers
# ====================
def print_stage(title: str) -> None:
    bar = "=" * 52
    print(f"\n{bar}")
    print(f"[ {title} ]")
    print(bar)


# ====================
# Environment Factory
# ====================
def make_env() -> gym.Env:
    env = FullyObsWrapper(gym.make("BabyAI-OneRoomS8-v0"))
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    return env


# ====================
# Rollout Helper
# ====================
def _rollout_one_episode(
    collector: minari.DataCollector,
    seed: int,
    max_steps: int = 1024,
) -> int:
    obs, _ = collector.reset(seed=seed)

    steps = 0
    for _ in range(max_steps):
        action = collector.action_space.sample()
        try:
            obs, _, terminated, truncated, _ = collector.step(action)
        except Exception:
            print(f"step_exception_at={steps + 1} action={action}")
            raise

        _ = obs
        print(f"step={steps + 1} action={action}")

        steps += 1
        if terminated or truncated:
            break
    return steps


# ====================
# Script Main
# ====================
dataset_id = "oneroom-S8-fullobs-test-v0"

# ---- Init Collector ----
print_stage("Init Collector")
collector = minari.DataCollector(make_env(), record_infos=False)

# ---- API: reset + step ----
print_stage("API: reset + step")
steps_0 = _rollout_one_episode(collector, seed=55)
print(f"episode_steps={steps_0}")

# ---- API: create_dataset ----
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
    eval_env=make_env(),
    description="DataCollector API smoke script",
)
print(f"created_dataset_id={dataset.spec.dataset_id}")
print(f"created_total_episodes={dataset.total_episodes}")
print(f"created_total_steps={dataset.total_steps}")

# ---- API: add_to_dataset ----
print_stage("API: add_to_dataset")
extra_collector = minari.DataCollector(make_env(), record_infos=False)
steps_1 = _rollout_one_episode(extra_collector, seed=99)
print(f"extra_episode_steps={steps_1}")

extra_collector.add_to_dataset(dataset)
print(f"after_add_total_episodes={dataset.total_episodes}")
print(f"after_add_total_steps={dataset.total_steps}")

# ---- API: close ----
print_stage("API: close")
extra_collector.close()
collector.close()

# ---- Cleanup ----
print_stage("Cleanup")
minari.delete_dataset(dataset_id)

print_stage("Done")
