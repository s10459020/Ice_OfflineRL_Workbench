from __future__ import annotations

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
