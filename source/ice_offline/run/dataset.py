import json
from pathlib import Path

from ice_offline.dataset.base import Dataset


def eval_dataset(
    dataset: Dataset,
    returns_output_path: Path,
    steps_output_path: Path,
) -> tuple[Path, Path]:
    returns = [float(episode.rewards.sum()) for episode in dataset.episodes]
    steps = [float(len(episode.rewards)) for episode in dataset.episodes]

    returns_output_path.parent.mkdir(parents=True, exist_ok=True)
    with returns_output_path.open("w", encoding="utf-8") as file:
        json.dump(returns, file)

    steps_output_path.parent.mkdir(parents=True, exist_ok=True)
    with steps_output_path.open("w", encoding="utf-8") as file:
        json.dump(steps, file)

    print(f"saved: {returns_output_path}")
    print(f"saved: {steps_output_path}")
    return returns_output_path, steps_output_path
