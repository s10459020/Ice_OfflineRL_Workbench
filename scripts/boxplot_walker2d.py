from pathlib import Path

from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.boxplot import boxplot_data


DATASETS = [
    ("Random", "walker2d_random"),
    ("Medium", "walker2d_d4rl_medium"),
    ("Hybrid", "walker2d_d4rl_hybrid"),
    ("Expert", "walker2d_d4rl_expert"),
    ("Replay-Medium", "walker2d_replay_medium"),
    ("Replay-Expert", "walker2d_replay_expert"),
]


def trajectory_returns(dataset_id: str) -> list[float]:
    dataset = make_dataset(dataset_id, device="cpu")
    values = [float(episode.rewards.sum()) for episode in dataset.episodes]
    print(f"{dataset_id}: trajectories={len(values)}, transitions={dataset.count}")
    return values


def save_boxplot() -> Path:
    labels = [label for label, _ in DATASETS]
    values = [trajectory_returns(dataset_id) for _, dataset_id in DATASETS]
    output_path = Path(__file__).resolve().parents[1] / "documents" / "boxplot_result" / "walker2d.png"
    boxplot_data(
        "Walker2d Dataset Trajectory Returns",
        labels,
        values,
        output_path,
    )
    return output_path


if __name__ == "__main__":
    path = save_boxplot()
    print(f"saved: {path}")
