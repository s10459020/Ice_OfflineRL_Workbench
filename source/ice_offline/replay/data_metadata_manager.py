
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DataMetadataManager:
    """Unified metadata read/write/build interface."""

    DEFAULT_ENV_ID = "BabyAI-OneRoomS8-v0"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, metadata: dict[str, Any]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.path

    @classmethod
    def build(
        cls,
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
            "env_spec": json.dumps({"id": str(env_id)}, ensure_ascii=True),
        }
        if extra:
            metadata.update(extra)
        return metadata

    @classmethod
    def resolve_env_id(cls, metadata: dict[str, Any], default_env_id: str = DEFAULT_ENV_ID) -> str:
        raw = metadata.get("env_spec")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                env_id = parsed.get("id")
                if isinstance(env_id, str) and env_id:
                    return env_id
        return default_env_id
