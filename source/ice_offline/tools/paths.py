import os
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_root() -> Path:
    root = os.getenv("MINARI_DATASETS_PATH")
    if root:
        return Path(root)
    return Path("tmps/datasets")


def model_root() -> Path:
    return Path("tmps/model")


def eval_root() -> Path:
    return Path("tmps/eval")


def eval_plot_root() -> Path:
    return ensure_dir(eval_root() / "plots")
