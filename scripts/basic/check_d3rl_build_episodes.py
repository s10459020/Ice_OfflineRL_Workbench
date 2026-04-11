import numpy as np
import d3rlpy

# mute logging
d3rlpy.logging.LOG.info = lambda *args, **kwargs: None

DEVICE = "cuda:0"


def build_dummy_episodes() -> list[d3rlpy.dataset.Episode]:
    # episode 0: true terminal at the end => transition_count == size
    ep0_len = 5
    ep0 = d3rlpy.dataset.Episode(
        observations=np.random.random((ep0_len, 5)).astype(np.float32),
        actions=np.random.random((ep0_len, 2)).astype(np.float32),
        rewards=np.random.random((ep0_len, 1)).astype(np.float32),
        terminated=True,
    )

    # episode 1: timeout/truncation style => terminated=False => transition_count == size - 1
    ep1_len = 5
    ep1 = d3rlpy.dataset.Episode(
        observations=np.random.random((ep1_len, 5)).astype(np.float32),
        actions=np.random.random((ep1_len, 2)).astype(np.float32),
        rewards=np.random.random((ep1_len, 1)).astype(np.float32),
        terminated=False,
    )

    return [ep0, ep1]


def build_replay_buffer_from_episodes(
    episodes: list[d3rlpy.dataset.Episode],
) -> d3rlpy.dataset.ReplayBuffer:
    return d3rlpy.dataset.create_infinite_replay_buffer(episodes=episodes)


def main() -> None:
    episodes = build_dummy_episodes()

    print("episode_count:", len(episodes))
    for i, ep in enumerate(episodes):
        print(
            f"episode={i} size={ep.size()} terminated={ep.terminated} "
            f"transition_count={ep.transition_count}"
        )

    dataset = build_replay_buffer_from_episodes(episodes)

    print("dataset_type:", type(dataset))
    print("dataset.transition_count:", dataset.transition_count)
    print("dataset.action_space:", dataset.dataset_info.action_space)
    print("dataset.action_size:", dataset.dataset_info.action_size)

    # Use SAC for continuous actions.
    algo = d3rlpy.algos.SACConfig(batch_size=4).create(device=DEVICE)
    algo.fit(dataset, n_steps=max(10, dataset.transition_count), n_steps_per_epoch=10)
    print("fit finished")


if __name__ == "__main__":
    main()
