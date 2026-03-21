import gymnasium as gym
import minigrid  # noqa: F401
import minari

from ice_offline.strategy import collector
from ice_offline.tools import print_stage


# ====================
# Script Main
# ====================
# ---- Collect ----
print_stage("Collect")

env = gym.make("BabyAI-OneRoomS8-v0")
policy = lambda _obs: int(env.action_space.sample())

steps = collector.run(
    env=env,
    policy=policy,
    dataset_id="test_collector-v0",
    max_episodes=10,
    seed=123,
    overwrite=True,
)
print(f"collect_steps={steps}")

# ---- Verify dataset ----
print_stage("Verify Dataset")
dataset = minari.load_dataset("test_collect-v0")
print(f"dataset_id={dataset.spec.dataset_id}")
print(f"total_episodes={dataset.total_episodes}")
print(f"total_steps={dataset.total_steps}")
