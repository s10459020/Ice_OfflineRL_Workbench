from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import minari
from replay import StateDatasetReader, StateDatasetWriter, convert_observation
from replay.read_metadata import read_metadata, resolve_env_id
from minigrid.wrappers import FullyObsWrapper
from tools import stage

from strategy import replay

metadata_path = Path("tmps/metadata.json")
state_dataset_path = Path("tmps/main_data_info_converted.hdf5")
max_episodes: int | None = None

###############################################################################
# STAGE: LOAD
###############################################################################
stage("load")
if not metadata_path.exists():
    raise FileNotFoundError(f"metadata not found: {metadata_path}")

metadata = read_metadata(metadata_path)
dataset_id = str(metadata.get("dataset_id", "unknown"))
env_id = resolve_env_id(metadata)
dataset = minari.load_dataset(dataset_id)
selected_count = len(dataset) if max_episodes is None else max(0, min(int(max_episodes), len(dataset)))
print(
    "load_done "
    f"| dataset_id={dataset_id} "
    f"| env_id={env_id} "
    f"| source_episodes={len(dataset)} "
    f"| selected_episodes={selected_count}"
)

###############################################################################
# STAGE: CONVERT
###############################################################################
stage("convert")
writer = StateDatasetWriter(output_path=state_dataset_path, flush_interval=1)
try:
    for episode_index in range(selected_count):
        observations = dataset[episode_index].observations
        states = convert_observation(observations)
        writer.push_episode(states)
        print(
            f"convert_episode source=minari_episode_{episode_index} "
            f"target=episode_{episode_index} "
            f"num_states={len(states)}"
        )
    writer.flush()
finally:
    writer.close()
convert_result = {"path": str(state_dataset_path), "converted_episodes": selected_count}

print(
    "convert_done "
    f"| path={convert_result['path']} "
    f"| converted_episodes={convert_result['converted_episodes']}"
)

###############################################################################
# STAGE: REPLAY
###############################################################################
stage("replay")
replay_env = gym.make(env_id, render_mode="human")
replay_env = FullyObsWrapper(replay_env)
with StateDatasetReader(state_dataset_path) as reader:
    replay_episodes = replay(
        env=replay_env,
        reader=reader,
        episodes=selected_count,
        render_flag=True,
        print_flag=True,
    )

print(f"replay_done | episodes={len(replay_episodes)} | path={state_dataset_path}")
for item in replay_episodes:
    print(
        f"replay_episode={item['episode_index']} "
        f"states={item['num_states']}"
    )
