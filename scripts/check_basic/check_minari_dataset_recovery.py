import minari

from ice_offline.tools import print_stage

def step_load_dataset(dataset_id: str) -> minari.MinariDataset:
    print_stage("Load Dataset")
    print(f"dataset_id={dataset_id}")
    return minari.load_dataset(dataset_id)


def step_recover_environment(dataset: minari.MinariDataset) -> None:
    print_stage("API: recover_environment")
    recovered_env = dataset.recover_environment(eval_env=True)
    print(f"recovered_env_type={type(recovered_env).__name__}")
    recovered_env.close()


def main() -> None:
    dataset_id = "minari_collector-v0"
    dataset = step_load_dataset(dataset_id)
    step_recover_environment(dataset)
    print_stage("Done")


if __name__ == "__main__":
    main()
