import gymnasium as gym
import minigrid  # noqa: F401
import minari
import numpy as np
from collections import defaultdict

from ice_offline.env.common import MissionTextWrapper, NoJpegImageWrapper
from ice_offline.env.replay import StateRecordWrapper, ValueRecordWrapper
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "test_collect_value-v0"
MAX_EPISODES = 10

def value_fn(obs, action: int, set_value: float | None = None) -> float:
    key = (obs["image"].tobytes(), obs["direction"])
    if set_value is not None:
        value_table[key][action] = set_value
    return value_table[key][action]


# ====================
# Collect
# ====================
print_stage("Collect")

env = gym.make("BabyAI-OneRoomS8-v0")
env = MissionTextWrapper(env)
env = NoJpegImageWrapper(env)
env = StateRecordWrapper(env)
env = ValueRecordWrapper(env)
collector = minari.DataCollector(env, record_infos=True)
eval_env = gym.make("BabyAI-OneRoomS8-v0")
steps = 0

ACTION_DIM = collector.action_space.n
value_table: defaultdict[tuple[bytes, int], np.ndarray] = defaultdict(
    lambda: np.zeros(ACTION_DIM, dtype=np.float32)
)

try:
    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = collector.reset()
        values = env.record(value_fn)
        episode_steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = int(np.random.randint(0, 4))
            obs, _, done, truncated, _ = collector.step(action)
            episode_steps += 1
            steps += 1
            value = value_fn(obs, action, set_value=steps)
            values = env.record(value_fn)
            print(
                f"episode={episode} step={episode_steps} "
                f"global_steps={steps} action={action} value={value} values_max={np.max(values):.1f}"
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
        code_permalink="https://example.com/test_collect_value",
        eval_env=eval_env,
        description="collect value-info dataset smoke script",
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
episode0 = dataset[0]
values = episode0.infos["values"]
print(f"dataset_id={dataset.spec.dataset_id}")
print(f"total_episodes={dataset.total_episodes}")
print(f"total_steps={dataset.total_steps}")
print(f"values_shape={np.asarray(values).shape}")
