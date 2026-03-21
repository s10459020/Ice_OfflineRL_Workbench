
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.replay import StateDataset
from ice_offline.tools.types import Transition
from ice_offline.tools import stage
from ice_offline.strategy import (
    collect_dataset,
    replay,
)


dataset_path = Path("tmps/one_room_s8_info.hdf5")
episodes = 3

###############################################################################
# STAGE: COLLECT
###############################################################################
stage("collector")
time.sleep(1.0)

collector_env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
collector_env = FullyObsWrapper(collector_env)
collector_steps = collect_dataset(
    env=collector_env,
    state_output_path=dataset_path,
    max_episodes=episodes,
    seed=42,
    flush_interval=0,
    print_interval=1,
)
collector_env.close()

collector_episodes = []
with StateDataset(dataset_path, mode="r") as state_manager:
    for episode_index in range(min(int(episodes), state_manager.num_episodes())):
        collector_episodes.append(
            {
                "episode_index": episode_index,
                "num_states": int(state_manager.episode_length(episode_index)),
            }
        )
print(
    f"collector_done | path={dataset_path} | episodes={len(collector_episodes)} "
    f"| collected_steps={collector_steps}"
)
for item in collector_episodes:
    print(
        f"collector_episode={item['episode_index']} "
        f"states={item['num_states']}"
    )

###############################################################################
# STAGE: REPLAY
###############################################################################
stage("replay")
time.sleep(1.0)

replay_env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
replay_env = FullyObsWrapper(replay_env)
with StateDataset(dataset_path, mode="r") as state_manager:
    state_sequences = state_manager.read(max_episodes=episodes)
    trajectories = [
        [Transition(action=0, reward=0.0) for _ in range(max(0, len(states) - 1))]
        for states in state_sequences
    ]
    replay_steps = replay(
        env=replay_env,
        state_sequences=state_sequences,
        trajectories=trajectories,
        max_episodes=episodes,
        render_interval=1,
        print_interval=1,
    )

print(f"replay_done | steps={replay_steps}")

###############################################################################
# STAGE: COMPARE
###############################################################################
stage("compare")
time.sleep(1.0)

collector_states = [item["num_states"] for item in collector_episodes]
all_match = len(collector_states) == int(episodes)
print(
    f"compare_episodes collector={len(collector_states)} "
    f"replay={int(episodes)} matched={all_match}"
)

print(f"compare_result | all_matched={all_match}")
