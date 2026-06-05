import numpy as np
import torch

from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset


BATCH_SIZE = 4


def _check_metadata(dataset: HopperMediumD4rlDataset) -> None:
    print("=== Metadata ===")
    print(f"id={dataset.id}")
    print(f"env_id={dataset.env_id}")
    print(f"obs_shape={dataset.obs_shape}")
    print(f"act_shape={dataset.act_shape}")
    print(f"obs_dim={dataset.obs_dim}")
    print(f"act_dim={dataset.act_dim}")
    print(f"count={dataset.count}")

    if dataset.obs_shape != (11,):
        raise SystemExit(f"FAIL: obs_shape expected=(11,) actual={dataset.obs_shape}")
    if dataset.act_shape != (3,):
        raise SystemExit(f"FAIL: act_shape expected=(3,) actual={dataset.act_shape}")
    if dataset.count != 1000000:
        raise SystemExit(f"FAIL: count expected=1000000 actual={dataset.count}")


def _check_buffer(dataset: HopperMediumD4rlDataset) -> None:
    print("=== Buffer ===")
    buffer = dataset.buffer
    print(f"observations={tuple(buffer.observations.shape)}")
    print(f"next_observations={tuple(buffer.next_observations.shape)}")
    print(f"actions={tuple(buffer.actions.shape)}")
    print(f"rewards={tuple(buffer.rewards.shape)}")
    print(f"dones={tuple(buffer.dones.shape)}")

    if tuple(buffer.observations.shape) != (dataset.count, *dataset.obs_shape):
        raise SystemExit("FAIL: buffer observations shape mismatch")
    if tuple(buffer.next_observations.shape) != (dataset.count, *dataset.obs_shape):
        raise SystemExit("FAIL: buffer next_observations shape mismatch")
    if tuple(buffer.actions.shape) != (dataset.count, *dataset.act_shape):
        raise SystemExit("FAIL: buffer actions shape mismatch")
    if tuple(buffer.rewards.shape) != (dataset.count,):
        raise SystemExit("FAIL: buffer rewards shape mismatch")
    if tuple(buffer.dones.shape) != (dataset.count,):
        raise SystemExit("FAIL: buffer dones shape mismatch")


def _check_sample_batch(dataset: HopperMediumD4rlDataset) -> None:
    print("=== Sample Batch ===")
    observations, actions, rewards, next_observations, dones = dataset.sample_batch(BATCH_SIZE)
    print(f"observations={tuple(observations.shape)}")
    print(f"actions={tuple(actions.shape)}")
    print(f"rewards={tuple(rewards.shape)}")
    print(f"dones={tuple(dones.shape)}")
    print(f"next_observations={tuple(next_observations.shape)}")

    if tuple(observations.shape) != (BATCH_SIZE, *dataset.obs_shape):
        raise SystemExit("FAIL: sampled observations shape mismatch")
    if tuple(actions.shape) != (BATCH_SIZE, *dataset.act_shape):
        raise SystemExit("FAIL: sampled actions shape mismatch")
    if tuple(rewards.shape) != (BATCH_SIZE,):
        raise SystemExit("FAIL: sampled rewards shape mismatch")
    if tuple(dones.shape) != (BATCH_SIZE,):
        raise SystemExit("FAIL: sampled dones shape mismatch")
    if tuple(next_observations.shape) != (BATCH_SIZE, *dataset.obs_shape):
        raise SystemExit("FAIL: sampled next_observations shape mismatch")


def _check_episodes(dataset: HopperMediumD4rlDataset) -> None:
    print("=== Episodes ===")
    episodes = dataset.episodes
    transition_count = sum(int(episode.actions.shape[0]) for episode in episodes)
    print(f"episode_count={len(episodes)}")
    print(f"transition_count={transition_count}")

    if transition_count != dataset.count:
        raise SystemExit(f"FAIL: episode transition_count expected={dataset.count} actual={transition_count}")
    if len(episodes) != dataset.episode_count:
        raise SystemExit(f"FAIL: episode_count expected={dataset.episode_count} actual={len(episodes)}")
    if [int(episode.actions.shape[0]) for episode in episodes] != dataset.step_counts:
        raise SystemExit("FAIL: step_counts mismatch")

    for episode_index, episode in enumerate(episodes):
        step_count = int(episode.actions.shape[0])
        if episode.observations.shape[0] != step_count + 1:
            raise SystemExit(f"FAIL: episode observations length mismatch episode={episode_index}")
        if episode.rewards.shape[0] != step_count:
            raise SystemExit(f"FAIL: episode rewards length mismatch episode={episode_index}")
        if episode.terminations.shape[0] != step_count:
            raise SystemExit(f"FAIL: episode terminations length mismatch episode={episode_index}")
        if episode.truncations.shape[0] != step_count:
            raise SystemExit(f"FAIL: episode truncations length mismatch episode={episode_index}")

    first_episode = episodes[0]
    first_step_count = int(first_episode.actions.shape[0])
    buffer = dataset.buffer
    first_final_observation = first_episode.observations[-1]
    expected_final_observation = buffer.next_observations[first_step_count - 1].cpu().numpy()
    if not np.array_equal(first_final_observation, expected_final_observation):
        raise SystemExit("FAIL: first episode final observation does not match buffer next_observations")

    if len(episodes) > 1:
        second_initial_observation = episodes[1].observations[0]
        first_terminal_observation = first_episode.observations[-1]
        if np.array_equal(second_initial_observation, first_terminal_observation):
            print("WARN: second episode initial observation equals first terminal observation")


def main() -> None:
    torch.manual_seed(0)
    dataset = HopperMediumD4rlDataset()
    _check_metadata(dataset)
    _check_buffer(dataset)
    _check_sample_batch(dataset)
    _check_episodes(dataset)
    print("PASS: d4rl_loader")


if __name__ == "__main__":
    main()
