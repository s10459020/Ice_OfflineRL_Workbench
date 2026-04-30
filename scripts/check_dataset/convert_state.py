import gymnasium as gym
import minigrid  # noqa: F401

from ice_offline.dataset import StateInjectWrapper, convert_fullobs
from ice_offline.env.common import insert_render_quiet_innermost
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
ENV_ID = "BabyAI-OneRoomS8-v0"


# ====================
# Convert
# ====================
print_stage("Convert")
state_data_path = convert_fullobs(dataset_source_id=DATASET_ID)
print(f"state_data_path={state_data_path}")


# ====================
# Replay (Human)
# ====================
print_stage("Replay Human")
base_env = gym.make(ENV_ID, render_mode="human")
base_env = insert_render_quiet_innermost(base_env)
env = StateInjectWrapper(base_env, dataset_id=DATASET_ID, random_episode=False)

steps_online = 0
for episode in range(1, env.total_episodes + 1):
    env.reset()
    env.render()

    episode_step = 0
    while True:
        _, reward, terminated, truncated, info = env.step(None)
        episode_step += 1
        steps_online += 1

        env.render()
        print(
            f"step={steps_online} episode={episode} episode_step={episode_step} "
            f"action={info.get('action')} reward={reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )

        if terminated or truncated:
            break

env.close()
print(f"online_steps={steps_online}")
