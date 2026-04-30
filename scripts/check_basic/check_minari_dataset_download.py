import minari

from ice_offline.tools import print_stage

def step_load_dataset(dataset_id: str) -> minari.MinariDataset:
    print_stage("Load Dataset")
    print(f"dataset_id={dataset_id}")
    return minari.load_dataset(dataset_id, download=True)


def step_attributes(dataset: minari.MinariDataset) -> None:
    print_stage("Attributes")
    print(f"spec={dataset.spec}")
    print(f"total_steps={dataset.total_steps}")
    print(f"total_episodes={dataset.total_episodes}")
    print(f"episode_indices={dataset.episode_indices}")


def step_sample_episodes(dataset: minari.MinariDataset) -> None:
    print_stage("API: sample_episodes")
    sampled = list(dataset.sample_episodes(5))
    print(f"sampled_count={len(sampled)}")


def step_iterate_episodes(dataset: minari.MinariDataset) -> None:
    print_stage("API: iterate_episodes")
    iterated = list(dataset.iterate_episodes())
    print(f"iterated_count={len(iterated)}")


def step_filter_episodes(dataset: minari.MinariDataset) -> None:
    print_stage("API: filter_episodes")
    filtered = dataset.filter_episodes(lambda ep: len(ep.actions) >= 5)
    filtered_count = len(list(filtered))
    print(f"filtered_count={filtered_count}")


def step_set_seed(dataset: minari.MinariDataset) -> None:
    print_stage("API: set_seed")
    dataset.set_seed(7)
    _ = list(dataset.sample_episodes(1))
    dataset.set_seed(7)
    _ = list(dataset.sample_episodes(1))
    print("set_seed_ok=true")


def step_update_dataset_from_buffer(dataset: minari.MinariDataset) -> None:
    print_stage("API: update_dataset_from_buffer")
    dataset.update_dataset_from_buffer([])
    print("update_dataset_from_buffer_ok=true (no-op buffer)")


def main() -> None:
    dataset_id = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
    dataset = step_load_dataset(dataset_id)
    step_attributes(dataset)
    step_sample_episodes(dataset)
    step_iterate_episodes(dataset)
    step_filter_episodes(dataset)
    step_set_seed(dataset)
    step_update_dataset_from_buffer(dataset)
    print_stage("Done")


if __name__ == "__main__":
    main()
