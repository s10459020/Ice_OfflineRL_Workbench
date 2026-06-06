from pathlib import Path

from ice_offline.config.paths import DATASETS_ROOT


def skip_missing(path: Path) -> bool:
    if not (DATASETS_ROOT / path).exists():
        print(f"skip missing: {path}")
        return True
    return False


