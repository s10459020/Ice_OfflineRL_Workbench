from pathlib import Path

from ice_offline.tools.paths import dataset_root


SKIP = True


def data_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.exists() or path_obj.as_posix().startswith("tmps/"):
        return path_obj
    return dataset_root() / path_obj


def data_exists(path: str) -> bool:
    return data_path(path).exists()


def skip_missing(label: str, path: str | Path) -> bool:
    if SKIP and not Path(path).exists():
        print(f"skip missing: {label}")
        return True
    return False


def skip_missing_data(path: str) -> bool:
    if SKIP and not data_exists(path):
        print(f"skip missing: {path}")
        return True
    return False


def skip_existing(label: str, path: str | Path) -> bool:
    if SKIP and Path(path).exists():
        print(f"skip existing: {label}")
        return True
    return False
