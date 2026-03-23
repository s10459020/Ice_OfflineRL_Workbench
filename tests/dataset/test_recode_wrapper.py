import gymnasium as gym
import minigrid  # noqa: F401

import minari
from minigrid.wrappers import FullyObsWrapper

from ice_offline.replay import StateRecordWrapper
from ice_offline.strategy import online
from ice_offline.tools import MissionTextWrapper, NoJpegImageWrapper, print_stage


# ====================
# Environment Factory
# ====================
def make_env() -> gym.Env:
    env = FullyObsWrapper(gym.make("BabyAI-OneRoomS8-v0"))
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    env = StateRecordWrapper(env)
    return env


# ====================
# Script Main
# ====================
# ---- Init Collector ----
print_stage("Init Collector")
collector = minari.DataCollector(make_env(), record_infos=True)

# ---- Rollout ----
print_stage("Rollout")
steps = online.test(
    collector,
    policy=lambda _obs: int(collector.action_space.sample()),
    max_episodes=3,
    seed=123,
)
print(f"episode_steps={steps}")

# ---- Create Dataset ----
print_stage("Create Dataset")
try:
    minari.delete_dataset("test_recode_wrapper-v0")
except Exception:
    pass

dataset = collector.create_dataset(
    dataset_id="test_recode_wrapper-v0",
    algorithm_name="random",
    author="local_test",
    author_email="local_test@example.com",
    code_permalink="https://example.com/oneroom-s8-capture-test",
    eval_env=make_env(),
    description="StateCapture + Minari collector script",
)
print(f"created_dataset_id={dataset.spec.dataset_id}")
print(f"created_total_episodes={dataset.total_episodes}")
print(f"created_total_steps={dataset.total_steps}")

# ---- Close ----
print_stage("Close")
collector.close()

# ---- Cleanup ----
print_stage("Cleanup")
#minari.delete_dataset(dataset_id)

print_stage("Done")
