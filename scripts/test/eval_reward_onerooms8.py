import os
from pathlib import Path

import minari
import numpy as np

from ice_offline.tools.paths import eval_root
from ice_offline.tools.paths import minari_root


DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
RUNNER_ID = "onerooms8"
# reward_sum in current training logs is sum(r), equivalent to discounted return with gamma=1.0.
GAMMA = 1.0
QUANTILES = [0.10, 0.25, 0.50, 0.75]


def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += float(reward) * discount
        discount *= gamma
    return float(total)


def main() -> None:
    os.environ.setdefault("MINARI_DATASETS_PATH", str(minari_root()))
    dataset = minari.load_dataset(DATASET_ID, download=True)
    returns: list[float] = []
    for episode in dataset.iterate_episodes():
        rewards = np.asarray(episode.rewards, dtype=np.float64)
        returns.append(discounted_return(rewards, gamma=GAMMA))

    if not returns:
        raise RuntimeError(f"no episodes found in dataset: {DATASET_ID}")

    returns_array = np.asarray(returns, dtype=np.float64)
    quantile_values = np.quantile(returns_array, QUANTILES)
    output_path = Path(eval_root()) / f"eval_reward_{RUNNER_ID}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"dataset_id={DATASET_ID}",
        f"gamma={GAMMA}",
        f"episodes={len(returns)}",
        f"reward_mean={returns_array.mean():.6f}",
    ]
    for q, value in zip(QUANTILES, quantile_values):
        lines.append(f"reward_q{int(q * 100):02d}={float(value):.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
