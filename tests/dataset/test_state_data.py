import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from ice_offline.dataset.state_collector import StateCollector
from ice_offline.dataset.state_loader import StateLoader
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "test_state_data-v0"
MAX_EPISODES = 10



# ====================
# Build State Data
# ====================
print_stage("Build State Data")
env = gym.make("BabyAI-OneRoomS8-v0")
collector = StateCollector(env.unwrapped)
steps = 0

try:
    for episode in range(1, MAX_EPISODES + 1):
        collector.reset(seed=episode)

        done = False
        truncated = False
        ep_steps = 0
        while not (done or truncated):
            action = int(np.random.randint(0, 4))
            _, _, done, truncated, _ = collector.step(action)
            steps += 1
            ep_steps += 1
            print(f"episode={episode} step={ep_steps} action={action} global_steps={steps}")

        print(f"episode={episode} end steps={ep_steps} done={done} truncated={truncated}")

    out_path = collector.save(DATASET_ID)
    print(f"state_data_path={out_path}")
finally:
    collector.close()



# ====================
# Load State Data
# ====================
print_stage("Load State Data")
loader = StateLoader(DATASET_ID)
try:
    episode_count = loader.get_episode_count()
    episode0 = loader.load_episode(0)
    step0 = loader.load_step(0, 0)
    print(f"episode_count={episode_count}")
    print(f"episode0_len={len(episode0)}")
    print(f"episode0_step0_type={type(episode0[0]).__name__}")
    print(f"step0_type={type(step0).__name__}")
    print(f"step0_agent_dir={step0.agent_dir}")
finally:
    loader.close()
