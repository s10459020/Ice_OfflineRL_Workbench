from pathlib import Path

from ice_offline.tools.paths import minari_root


SKIP = True


def data_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if path.exists() or dataset_path.startswith("tmps/"):
        return path
    return minari_root() / dataset_path


def data_exists(dataset_path: str) -> bool:
    return data_path(dataset_path).exists()


def skip_missing(label: str, path: str | Path) -> bool:
    if SKIP and not Path(path).exists():
        print(f"skip missing: {label}")
        return True
    return False


def skip_missing_data(dataset_path: str) -> bool:
    if SKIP and not data_exists(dataset_path):
        print(f"skip missing: {dataset_path}")
        return True
    return False


def skip_existing(label: str, path: str | Path) -> bool:
    if SKIP and Path(path).exists():
        print(f"skip existing: {label}")
        return True
    return False
