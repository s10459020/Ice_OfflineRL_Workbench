
import json
from pathlib import Path
from typing import Any


DEFAULT_ENV_ID = "BabyAI-OneRoomS8-v0"


def read_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_env_id(metadata: dict[str, Any], default_env_id: str = DEFAULT_ENV_ID) -> str:
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

