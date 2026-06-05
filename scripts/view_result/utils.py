from pathlib import Path

from ice_offline.tools.paths import dataset_root


def skip_missing(path: Path) -> bool:
    if not (dataset_root() / path).exists():
        print(f"skip missing: {path}")
        return True
    return False
