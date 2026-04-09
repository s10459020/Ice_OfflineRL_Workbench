import numpy as np
import d3rlpy

# mute logging
d3rlpy.logging.LOG.info = lambda *args, **kwargs: None

DEVICE = "cuda:0"

### terminals determine episode boundaries
def build_dummy_dataset() -> d3rlpy.dataset.MDPDataset:
    observations = np.random.random((20, 5)).astype(np.float32)
    actions = np.random.random((20, 2)).astype(np.float32)  # continuous action
    rewards = np.random.random(20).astype(np.float32)

    terminals = np.zeros(20, dtype=np.float32)
    terminals[4::5] = 1.0  # episode boundary every 5 steps

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    ### dataset ###
    print("terminal transition_count:", dataset.transition_count)

    ### dataset.dataset_info ###
    # print("action_space:", dataset.dataset_info.action_space)
    # print("action_size:", dataset.dataset_info.action_size)
    # print("action_signature:", dataset.dataset_info.action_signature)
    # print("observation_signature:", dataset.dataset_info.observation_signature)
    # print("reward_signature:", dataset.dataset_info.reward_signature)

    ### dataset.episodes ###
    print("episode_count:", len(dataset.episodes))
    for i, episode in enumerate(dataset.episodes):
        print(f"--- Episode {i} transition_count: {episode.transition_count} ---")
        # print("observation_signature:", episode.observation_signature)
        # print("observations:", episode.observations)
        # print("action_signature:", episode.action_signature)
        # print("actions:", episode.actions)
        # print("reward_signature:", episode.reward_signature)    
        # print("rewards:", episode.rewards)
        # print("terminated:", episode.terminated)

    return dataset


### terminals step is transition, but timeout step is not.
def build_dummy_timeout_dataset() -> d3rlpy.dataset.MDPDataset:
    observations = np.random.random((20, 5)).astype(np.float32)
    actions = np.random.random((20, 2)).astype(np.float32)  # continuous action
    rewards = np.random.random(20).astype(np.float32)

    terminals = np.array([1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0], dtype=np.float32)
    timeouts = np.zeros(20, dtype=np.float32)
    timeouts[4::5] = 1.0  # episode boundary every 5 steps

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    ### dataset ###
    print("timeout transition_count:", dataset.transition_count)

    ### dataset.dataset_info ###
    # print("action_space:", dataset.dataset_info.action_space)
    # print("action_size:", dataset.dataset_info.action_size)
    # print("action_signature:", dataset.dataset_info.action_signature)
    # print("observation_signature:", dataset.dataset_info.observation_signature)
    # print("reward_signature:", dataset.dataset_info.reward_signature)

    ### dataset.episodes ###
    print("episode_count:", len(dataset.episodes))
    for i, episode in enumerate(dataset.episodes):
        print(f"--- Episode {i} transition_count: {episode.transition_count} ---")
        # print("observation_signature:", episode.observation_signature)
        # print("observations:", episode.observations)
        # print("action_signature:", episode.action_signature)
        # print("actions:", episode.actions)
        # print("reward_signature:", episode.reward_signature)    
        print("rewards:", episode.rewards)
        print("terminated:", episode.terminated)

    return dataset


def main() -> None:
    dataset = build_dummy_dataset()
    dataset = build_dummy_timeout_dataset()

    # Use SAC for continuous actions.
    algo = d3rlpy.algos.SACConfig(batch_size=32).create(device=DEVICE)
    algo.fit(dataset, n_steps=dataset.transition_count)
    print("fit finished")


if __name__ == "__main__":
    main()