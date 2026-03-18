from __future__ import annotations

from pathlib import Path

from replay import StateDatasetReader


def _format_state(prefix: str, state) -> str:
    carrying = state.carrying if state.carrying is not None else "None"
    return (
        f"{prefix} "
        f"pos={state.agent_pos} "
        f"dir={state.agent_dir} "
        f"mission='{state.mission}' "
        f"carrying={carrying} "
        f"grid_shape={tuple(state.grid.shape)}"
    )


dataset_path = Path("tmps/one_room_s8_info.hdf5")
if not dataset_path.exists():
    raise FileNotFoundError(
        f"dataset not found: {dataset_path}. "
        "Run source/replay_scripts/test_state_writer.py first."
    )

with StateDatasetReader(dataset_path) as reader:
    print(
        "state_dataset_info "
        f"| path={dataset_path} "
        f"| num_episodes={reader.num_episodes} "
        f"| total_episodes_attr={reader.total_episodes}"
    )

    for episode_index in range(reader.num_episodes):
        length = reader.episode_length(episode_index)
        print(f"episode={episode_index} num_states={length}")
        for state_index in range(length):
            state = reader.get_state(episode_index=episode_index, state_index=state_index)
            print(_format_state(f"state[{state_index}]", state))
