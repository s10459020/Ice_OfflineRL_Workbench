import d3rlpy

from ice_offline.data.d3rl.converter import to_buffer
from ice_offline.tools.printer import print_stage


# ====================
# Config
# ====================
DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
DEVICE = "cuda:0"
N_STEPS = 200


# ====================
# Convert
# ====================
print_stage("Convert")
dataset = to_buffer(DATASET_ID)
print(f"dataset_type={type(dataset)}")
print(f"episode_count={len(dataset.episodes)}")
print(f"transition_count={dataset.transition_count}")
print(f"action_space={dataset.dataset_info.action_space}")
print(f"action_size={dataset.dataset_info.action_size}")


# ====================
# Train
# ====================
print_stage("Train")
algo = d3rlpy.algos.DQNConfig(batch_size=32).create(device=DEVICE)
algo.fit(dataset, n_steps=N_STEPS)
print("fit finished")


