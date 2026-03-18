from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from replay import StateDatasetReader
from strategy import collect_state_dataset_with_signatures, replay_state_dataset_with_signatures


def _banner(title: str) -> None:
    bar = "#" * 72
    print("\n" + bar)
    print(f"# {title.upper():^68} #")
    print(bar)


dataset_path = Path("tmps/one_room_s8_info.hdf5")
episodes = 3
max_episode_steps = 20
fps = 6

_banner("collector")
time.sleep(1.0)

collector_env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
collector_env = FullyObsWrapper(collector_env)
collector_result = collect_state_dataset_with_signatures(
    env=collector_env,
    output_path=dataset_path,
    max_episodes=episodes,
    max_episode_steps=max_episode_steps,
    seed=42,
    write_interval=0,
    fps=fps,
    print_flag=True,
)
collector_env.close()

collector_episodes = collector_result["episodes"]
print(f"collector_done | path={collector_result['path']} | episodes={len(collector_episodes)}")
for item in collector_episodes:
    print(
        f"collector_episode={item['episode_index']} "
        f"transitions={item['num_transitions']} "
        f"transition_code={item['transition_signature']} "
        f"state_code={item['state_signature']}"
    )

_banner("replay")
time.sleep(1.0)

with StateDatasetReader(dataset_path) as reader:
    replay_env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
    replay_env = FullyObsWrapper(replay_env)
    replay_episodes = replay_state_dataset_with_signatures(
        env=replay_env,
        reader=reader,
        episodes=episodes,
        fps=fps,
        reset_pause_sec=1.0,
        print_flag=True,
    )

print(f"replay_done | episodes={len(replay_episodes)}")
for item in replay_episodes:
    print(
        f"replay_episode={item['episode_index']} "
        f"states={item['num_states']} "
        f"state_code={item['state_signature']}"
    )

_banner("compare")
time.sleep(1.0)

collector_state_codes = [item["state_signature"] for item in collector_episodes]
replay_state_codes = [item["state_signature"] for item in replay_episodes]

all_match = len(collector_state_codes) == len(replay_state_codes)
for episode_index, (c_code, r_code) in enumerate(zip(collector_state_codes, replay_state_codes)):
    matched = c_code == r_code
    all_match = all_match and matched
    print(
        f"compare_episode={episode_index} "
        f"collector_state_code={c_code} "
        f"replay_state_code={r_code} "
        f"matched={matched}"
    )

print(f"compare_result | all_matched={all_match}")
