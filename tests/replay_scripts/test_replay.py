from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.replay import StateDatasetReader, serialize_state_trajectory
from ice_offline.tools import stage
from ice_offline.strategy import (
    collect_dataset,
    replay,
)


dataset_path = Path("tmps/one_room_s8_info.hdf5")
episodes = 3
max_episode_steps = 20

###############################################################################
# STAGE: COLLECT
###############################################################################
stage("collector")
time.sleep(1.0)

collector_env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
collector_env = FullyObsWrapper(collector_env)
collector_result = collect_dataset(
    env=collector_env,
    collect_state=True,
    collect_observation=False,
    state_output_path=dataset_path,
    max_episodes=episodes,
    max_episode_steps=max_episode_steps,
    seed=42,
    flush_interval=0,
    print_flag=True,
)
collector_env.close()

collector_episodes = []
with StateDatasetReader(dataset_path) as reader:
    for episode_index in range(min(int(episodes), reader.num_episodes)):
        states = list(reader.iter_episode_states(episode_index))
        serialized = serialize_state_trajectory(states, include_payload=False, include_signature=False)
        collector_episodes.append(
            {"episode_index": episode_index, "num_states": int(serialized["length"])}
        )
print(
    f"collector_done | path={collector_result['collect_state']['path']} | episodes={len(collector_episodes)}"
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
with StateDatasetReader(dataset_path) as reader:
    replay_episodes = replay(
        env=replay_env,
        reader=reader,
        episodes=episodes,
        render_flag=True,
        print_flag=True,
    )

print(f"replay_done | episodes={len(replay_episodes)}")
for item in replay_episodes:
    print(
        f"replay_episode={item['episode_index']} "
        f"states={item['num_states']}"
    )

###############################################################################
# STAGE: COMPARE
###############################################################################
stage("compare")
time.sleep(1.0)

collector_states = [item["num_states"] for item in collector_episodes]
replay_states = [item["num_states"] for item in replay_episodes]

all_match = len(collector_states) == len(replay_states)
for episode_index, (c_count, r_count) in enumerate(zip(collector_states, replay_states)):
    matched = c_count == r_count
    all_match = all_match and matched
    print(
        f"compare_episode={episode_index} "
        f"collector_states={c_count} "
        f"replay_states={r_count} "
        f"matched={matched}"
    )

print(f"compare_result | all_matched={all_match}")
