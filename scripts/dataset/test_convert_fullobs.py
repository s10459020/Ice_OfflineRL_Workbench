import minari

from ice_offline.dataset import converter
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
SOURCE_DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
TARGET_DATASET_ID = "test_convert_fullobs-v0"



# ====================
# Convert
# ====================
print_stage("Convert")
target_dataset = converter.convert_fullobs(
    dataset_source_id=SOURCE_DATASET_ID,
    dataset_target_id=TARGET_DATASET_ID,
)
print(f"target_dataset_id={target_dataset.spec.dataset_id}")
print(f"target_total_episodes={target_dataset.total_episodes}")
print(f"target_total_steps={target_dataset.total_steps}")


# ====================
# Verify 
# ====================
print_stage("Verify")
converted = minari.load_dataset(TARGET_DATASET_ID)
first_trajectory = converted[0]
infos = first_trajectory.infos
state_seq = infos["state"]
print(f"verified_dataset_id={converted.spec.dataset_id}")
print(f"verified_total_episodes={converted.total_episodes}")
print(f"verified_total_steps={converted.total_steps}")
print(f"first_trajectory_state_len={len(state_seq['mission'])}")
print(f"first_state_keys={list(state_seq.keys())}")
