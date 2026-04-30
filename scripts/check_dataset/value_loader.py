import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from collections import defaultdict

from ice_offline.env.replay import ValueCollector, ValueLoader
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "test_value_dataset-v0"
MAX_EPISODES = 10



# ====================
# Build Value Data
# ====================
print_stage("Build Value Data")

value_table: defaultdict[tuple[bytes, int], np.ndarray] = defaultdict(
    lambda: np.zeros(env.action_space.n, dtype=np.float32)
)

def value_fn(obs, action: int, set_value: float | None = None) -> float:
    key = (obs["image"].tobytes(), obs["direction"])
    if set_value is not None:
        value_table[key][action] = set_value
    return value_table[key][action]

env = gym.make("BabyAI-OneRoomS8-v0")
recorder = ValueCollector(env, value_fn)
steps = 0

try:
    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = recorder.reset(seed=episode)
        recorder.record()

        done = False
        truncated = False
        ep_steps = 0
        while not (done or truncated):
            action = int(np.random.randint(0, 4))
            prev_obs = obs
            obs, _, done, truncated, _ = recorder.step(action)
            steps += 1
            ep_steps += 1
            value_fn(prev_obs, action, set_value=steps)
            recorder.record()
            print(f"episode={episode} step={ep_steps} action={action} global_steps={steps}")
            
    out_path = recorder.save(DATASET_ID)
    print(f"value_data_path={out_path}")
finally:
    env.close()



# ====================
# Load Value Data
# ====================
print_stage("Load Value Data")
loader = ValueLoader(DATASET_ID)
try:
    episode_count = loader.get_episode_count()
    episode0 = loader.load_episode(0)
    step0 = loader.load_step(0, 0)
    print(f"episode_count={episode_count}")
    print(f"episode0_len={len(episode0)}")
    print(f"step0_shape={step0.shape}")
finally:
    loader.close()
