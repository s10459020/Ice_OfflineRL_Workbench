import gymnasium as gym
import minigrid  # noqa: F401
import minari

from ice_offline.strategy import collector


# ====================
# Script Main
# ====================
# ---- Config ----
dataset_id = "test_collect-v0"

# ---- Collect ----
env = gym.make("BabyAI-OneRoomS8-v0")
policy = lambda _obs: int(env.action_space.sample())
steps = collector.run(
    env=env,
    policy=policy,
    dataset_id=dataset_id,
    max_episodes=3,
    seed=123,
    overwrite=True,
)
print(f"collect_steps={steps}")

# ---- Verify dataset ----
dataset = minari.load_dataset(dataset_id)
print(f"dataset_id={dataset.spec.dataset_id}")
print(f"total_episodes={dataset.total_episodes}")
print(f"total_steps={dataset.total_steps}")
