import minari
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"


# ====================
# Script Main
# ====================
# ---- Load Dataset ----
print_stage("Load Dataset")
print(f"dataset_id={DATASET_ID}")
dataset = minari.load_dataset(DATASET_ID, download=True)

# ---- Attributes ----
print_stage("Attributes")
print(f"spec={dataset.spec}")
print(f"total_steps={dataset.total_steps}")
print(f"total_episodes={dataset.total_episodes}")
print(f"episode_indices={dataset.episode_indices}")

# ---- API: sample_episodes ----
print_stage("API: sample_episodes")
sampled = list(dataset.sample_episodes(5))
print(f"sampled_count={len(sampled)}")

# ---- API: iterate_episodes ----
print_stage("API: iterate_episodes")
iterated = list(dataset.iterate_episodes())
print(f"iterated_count={len(iterated)}")

# ---- API: filter_episodes ----
print_stage("API: filter_episodes")
filtered = dataset.filter_episodes(lambda ep: len(ep.actions) >= 5)
filtered_count = len(list(filtered))
print(f"filtered_count={filtered_count}")

# ---- API: set_seed ----
print_stage("API: set_seed")
dataset.set_seed(7)
_ = list(dataset.sample_episodes(1))
dataset.set_seed(7)
_ = list(dataset.sample_episodes(1))
print("set_seed_ok=true")

# ---- API: recover_environment ----
print_stage("API: recover_environment")
recovered_env = dataset.recover_environment()
print(f"recovered_env_type={type(recovered_env).__name__}")
recovered_env.close()

# ---- API: update_dataset_from_buffer ----
print_stage("API: update_dataset_from_buffer")
dataset.update_dataset_from_buffer([])
print("update_dataset_from_buffer_ok=true (no-op buffer)")

# ---- Done ----
print_stage("Done")
