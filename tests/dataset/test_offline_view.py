import minari

from ice_offline.tools import print_stage


# ====================
# Config
# ====================
DATASET_ID = "test_collect-v0"


# ====================
# Replay
# ====================
print_stage("Replay")
dataset = minari.load_dataset(DATASET_ID)
episode_indices = dataset.episode_indices

steps_seq = 0
for episode_no, trajectory_id in enumerate(episode_indices, start=1):
    trajectory = dataset[trajectory_id]
    for episode_step, (action, reward, terminated, truncated) in enumerate(
        zip(trajectory.actions, trajectory.rewards, trajectory.terminations, trajectory.truncations),
        start=1,
    ):
        steps_seq += 1
        print(
            f"step={steps_seq} episode={episode_no} trajectory_id={trajectory_id} episode_step={episode_step} "
            f"action={action} reward={reward:.3f} terminated={terminated} truncated={truncated}"
        )
print(f"sequential_steps={steps_seq}")
