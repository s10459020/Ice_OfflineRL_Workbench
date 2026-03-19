import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENV_ID = "BabyAI-OneRoomS8-v0"


def build_metadata(
    *,
    dataset_id: str,
    env_id: str = DEFAULT_ENV_ID,
    format_name: str = "observation_trajectory_v1",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_id": str(dataset_id),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": str(format_name),
        # Keep env_spec as JSON string for compatibility with resolve_env_id().
        "env_spec": json.dumps({"id": str(env_id)}, ensure_ascii=True),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path

