import gymnasium as gym
import minigrid  # noqa: F401
import minari

from ice_offline.env.common import MissionTextWrapper, NoJpegImageWrapper
from ice_offline.env.replay import StateRecordWrapper
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "test_collect_state-v0"
MAX_EPISODES = 10


# ====================
# Collect
# ====================
print_stage("Collect")
steps = 0

env = gym.make("BabyAI-OneRoomS8-v0")
env = MissionTextWrapper(env)
env = NoJpegImageWrapper(env)
env = StateRecordWrapper(env)
collector = minari.DataCollector(env, record_infos=True)
eval_env = gym.make("BabyAI-OneRoomS8-v0")

try:
    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = collector.reset()
        episode_steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = collector.action_space.sample()
            obs, _, done, truncated, _ = collector.step(action)
            _ = obs
            steps += 1
            episode_steps += 1
            print(
                f"episode={episode} step={episode_steps} "
                f"global_steps={steps} action={action} done={done} truncated={truncated}"
            )
        print(f"episode={episode} end episode_steps={episode_steps} done={done} truncated={truncated}")

    try:
        minari.delete_dataset(DATASET_ID)
    except Exception:
        pass

    collector.create_dataset(
        dataset_id=DATASET_ID,
        algorithm_name="random_policy",
        author="local_test",
        author_email="local_test@example.com",
        code_permalink="https://example.com/test_data_collect",
        eval_env=eval_env,
        description="collect smoke script",
    )
finally:
    eval_env.close()
    collector.close()

print(f"collect_steps={steps}")


# ====================
# Verify
# ====================
print_stage("Verify Dataset")
dataset = minari.load_dataset(DATASET_ID)
print(f"dataset_id={dataset.spec.dataset_id}")
print(f"total_episodes={dataset.total_episodes}")
print(f"total_steps={dataset.total_steps}")
