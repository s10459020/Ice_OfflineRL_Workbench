from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from replay.convert_fullobs import convert_fullobs
from replay.state_types import AgentState

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


def convert_minari_fullobs_dataset(
    dataset_id: str,
    output_path: str | Path,
    max_episodes: int | None = None,
    compression: str | None = "gzip",
) -> dict[str, Any]:
    """
    Convert a Minari full-observation dataset into state_dataset_v1.
    """
    if h5py is None:  # pragma: no cover
        raise ImportError("h5py is required for convert_minari_fullobs_dataset.") from _H5PY_IMPORT_ERROR

    try:
        import minari
    except ImportError as exc:  # pragma: no cover
        raise ImportError("minari is required for convert_minari_fullobs_dataset.") from exc

    dataset = minari.load_dataset(dataset_id)
    total_available = len(dataset)
    total_to_convert = total_available if max_episodes is None else max(0, min(int(max_episodes), total_available))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as file:
        file.attrs["format"] = "state_dataset_v1"
        file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        file.attrs["source"] = f"minari:{dataset_id}"
        file.attrs["total_episodes"] = int(total_to_convert)

        for episode_index in range(total_to_convert):
            states = convert_fullobs(dataset[episode_index])
            _write_episode(file=file, episode_index=episode_index, states=states, compression=compression)

    return {
        "dataset_id": dataset_id,
        "source_episodes": int(total_available),
        "converted_episodes": int(total_to_convert),
        "path": str(output),
    }


def _write_episode(
    file: Any,
    episode_index: int,
    states: list[AgentState],
    compression: str | None,
) -> None:
    episode_group = file.create_group(f"episode_{episode_index}")
    episode_group.attrs["num_states"] = len(states)
    utf8 = h5py.string_dtype(encoding="utf-8")

    missions = [str(state.mission) for state in states]
    positions = [(int(s.agent_pos[0]), int(s.agent_pos[1])) for s in states]
    directions = [int(s.agent_dir) for s in states]
    grids = [np.asarray(s.grid, dtype=np.uint8) for s in states]
    carrying = [json.dumps(s.carrying, ensure_ascii=True) for s in states]

    episode_group.create_dataset("mission", data=np.asarray(missions, dtype=object), dtype=utf8)
    episode_group.create_dataset("agent_pos", data=np.asarray(positions, dtype=np.int32))
    episode_group.create_dataset("agent_dir", data=np.asarray(directions, dtype=np.int8))
    episode_group.create_dataset("grid", data=np.asarray(grids, dtype=np.uint8), compression=compression)
    episode_group.create_dataset("carrying", data=np.asarray(carrying, dtype=object), dtype=utf8)
