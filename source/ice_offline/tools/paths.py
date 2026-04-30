
import os
from pathlib import Path


def minari_root() -> Path:
    root = os.getenv("MINARI_DATASETS_PATH")
    if root:
        return Path(root)
    return Path("tmps/datasets")


def model_root() -> Path:
    return Path("tmps/model")


def eval_root() -> Path:
    return Path("tmps/eval")


def resolve_value_data_path(dataset_id: str) -> Path:
    return minari_root() / dataset_id / "data" / "value_data.hdf5"
