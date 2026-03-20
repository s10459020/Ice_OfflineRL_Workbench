import minari


DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"


print(f"load_dataset id={DATASET_ID}")
dataset = minari.load_dataset(DATASET_ID, download=True)

print("attributes")
print(f"spec={dataset.spec}")
print(f"total_steps={dataset.total_steps}")
print(f"total_episodes={dataset.total_episodes}")
print(f"episode_indices={dataset.episode_indices}")

print("api: sample_episodes")
sampled = list(dataset.sample_episodes(5))
print(f"sampled_count={len(sampled)}")

print("api: iterate_episodes")
iterated = list(dataset.iterate_episodes())
print(f"iterated_count={len(iterated)}")

print("api: filter_episodes")
filtered = dataset.filter_episodes(lambda ep: len(ep.actions) >= 5)
try:
    filtered_count = len(filtered)  # type: ignore[arg-type]
except Exception:
    filtered_count = len(list(filtered))
print(f"filtered_count={filtered_count}")

print("api: set_seed")
dataset.set_seed(7)
_ = list(dataset.sample_episodes(1))
dataset.set_seed(7)
_ = list(dataset.sample_episodes(1))
print("set_seed_ok=true")

print("api: recover_environment")
recovered_env = dataset.recover_environment()
print(f"recovered_env_type={type(recovered_env).__name__}")
recovered_env.close()

print("api: update_dataset_from_buffer")
try:
    dataset.update_dataset_from_buffer([])
    print("update_dataset_from_buffer_ok=true (no-op buffer)")
except Exception as exc:
    print(f"update_dataset_from_buffer_ok=false reason={exc}")

print("done")
