import csv
from pathlib import Path

from generate_result_quick_tables import HYBRID_TABLES
from generate_result_quick_tables import REPRESENTATIVE_AGENTS
from generate_result_quick_tables import dataset_returns
from generate_result_quick_tables import formatted_cell
from generate_result_quick_tables import mean
from generate_result_quick_tables import scaled_score
from generate_result_quick_tables import top_score_threshold


RESULT_ROOT = Path(__file__).resolve().parent
HYBRID_VERSION_PATH = RESULT_ROOT / "agent_dataset_versions.csv"
OUTPUT_PATH = RESULT_ROOT / "hybrid_random_medium.csv"
LOWER_DATASET_ID = "walker2d_random"
UPPER_DATASET_ID = "walker2d_d4rl_medium"


def hybrid_version_records() -> dict[tuple[str, str], dict[str, str]]:
    with HYBRID_VERSION_PATH.open("r", encoding="utf-8", newline="") as file:
        return {
            (row["dataset"], row["agent"]): row
            for row in csv.DictReader(file)
            if row["experiment"] == "hybrid_random"
        }


def cell_suffix(cell: str) -> str:
    if "(" not in cell:
        return ""
    return cell.rsplit("(", 1)[1].split(")", 1)[0]


def generate() -> Path:
    records = hybrid_version_records()
    lower = mean(dataset_returns(LOWER_DATASET_ID))
    upper = mean(dataset_returns(UPPER_DATASET_ID))
    rows: list[list[str]] = []

    for dataset in HYBRID_TABLES:
        cells: list[tuple[float | None, str]] = []

        for agent in REPRESENTATIVE_AGENTS:
            record = records[(dataset.dataset_id, agent.agent_id)]
            raw_mean = float(record["raw_mean"]) if record["raw_mean"] else None
            if raw_mean is None:
                cells.append((None, record["cell"].rstrip("*")))
                continue

            score = scaled_score(raw_mean, lower, upper)
            suffix = cell_suffix(record["cell"])
            stale = record["cell"].startswith("!")
            cells.append((score, formatted_cell(score, suffix, stale)))

        scores = [score for score, _ in cells if score is not None]
        threshold = top_score_threshold(max(scores)) if scores else None
        rows.append([
            dataset.dataset_id,
            *[
                f"{cell}*" if score is not None and threshold is not None and score >= threshold else cell
                for score, cell in cells
            ],
        ])

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["task", *[agent.agent_id for agent in REPRESENTATIVE_AGENTS]])
        writer.writerows(rows)

    return OUTPUT_PATH


if __name__ == "__main__":
    path = generate()
    print(f"saved: {path}")
