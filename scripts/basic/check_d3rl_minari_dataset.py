import d3rlpy
import numpy as np
import minari

DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
DEVICE = "cuda:0"
N_STEPS = 200

def to_d3rl_dataset(n_steps: int = 1000) -> d3rlpy.dataset.MDPDataset:
    observations = np.random.random((n_steps, 100)).astype(np.float32)
    actions = np.random.random((n_steps, 4)).astype(np.float32)  # continuous action
    rewards = np.random.random(n_steps).astype(np.float32)

    terminals = np.zeros(n_steps, dtype=np.float32)
    timeouts = np.zeros(n_steps, dtype=np.float32)
    timeouts[199::200] = 1.0  # episode boundary every 200 steps

    return d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )


def main() -> None:
    minari_dataset = minari.load_dataset("minigrid-fourrooms-v0")
    dataset = to_d3rl_dataset(minari_dataset)

    print("transition_count:", dataset.transition_count)
    print("action_space:", dataset.dataset_info.action_space)
    print("action_size:", dataset.dataset_info.action_size)

    # Use SAC for continuous actions.
    algo = d3rlpy.algos.SACConfig(batch_size=32).create(device=DEVICE)
    algo.fit(dataset)
    print("fit finished")


if __name__ == "__main__":
    main()