import csv
from pathlib import Path

from _return import returns, returns_path
from _skip import data_exists


OUTPUT_ROOT = Path("tmps/returns")

AGENT_LIST = [
    "bc_deterministic",
    "bc_stochastic",
    "td3bc",
    "iql",
    "cql",
    "cql_max_q",
    "cql_soft_q",
    "aspl",
    "scas_mean",
    "scas_min",
]

DATASET_LIST = [
    ("random", "hopper_random"),
    ("replay", "hopper_replay"),
    ("expert_d4rl", "hopper_expert_d4rl"),
    ("medium_d4rl", "hopper_medium_d4rl"),
    ("medium_replay", "hopper_medium_replay"),
    ("medium_expert", "hopper_medium_expert"),
    ("simple", "hopper_simple"),
    ("medium", "hopper_medium"),
    ("expert", "hopper_expert"),
]

BOTTOM_LIST = [
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "test/hopper_simple-random-v0/data/main_data.hdf5",
    "test/hopper_medium-random-v0/data/main_data.hdf5",
    "test/hopper_expert-random-v0/data/main_data.hdf5",
]

TOP_LIST = [
    "",
    "tmps/datasets/d4rl/hopper_full_replay-v2.hdf5",
    "tmps/datasets/d4rl/hopper_expert-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium_replay-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium_expert-v2.hdf5",
    "mujoco/hopper/simple-v0/data/main_data.hdf5",
    "mujoco/hopper/medium-v0/data/main_data.hdf5",
    "mujoco/hopper/expert-v0/data/main_data.hdf5",
]


def source_exists(dataset_path: str) -> bool:
    if not dataset_path:
        return False
    if returns_path(dataset_path).exists():
        return True
    return data_exists(dataset_path)


def return_values(dataset_path: str) -> list[float] | None:
    if not source_exists(dataset_path):
        return None
    return returns(dataset_path)


def mean_return(dataset_path: str) -> float | None:
    values = return_values(dataset_path)
    if values is None:
        return None
    return sum(values) / len(values)


def max_return(dataset_path: str) -> float | None:
    values = return_values(dataset_path)
    if values is None:
        return None
    return max(values)


def cell(value: float | None) -> str:
    if value is None:
        return ""
    return str(float(value))


def scale(value: float | None, bottom: float | None, top: float | None) -> float | None:
    if value is None or bottom is None or top is None:
        return None
    if top == bottom:
        return None
    return (value - bottom) / (top - bottom) * 100.0


def actual_row(
    group_name: str,
    bottom_path: str,
    dataset_id: str,
    top_path: str,
) -> list[str]:
    values = [mean_return(bottom_path)]
    values.extend(mean_return(f"test/{dataset_id}-{agent_id}-v0/data/main_data.hdf5") for agent_id in AGENT_LIST)
    values.append(mean_return(top_path))
    return [group_name, *[cell(value) for value in values]]


def normalized_row(
    group_name: str,
    bottom_path: str,
    dataset_id: str,
    top_path: str,
    *,
    use_max_top: bool,
) -> list[str]:
    bottom = mean_return(bottom_path)
    top = max_return(top_path) if use_max_top else mean_return(top_path)
    values = [
        scale(mean_return(f"test/{dataset_id}-{agent_id}-v0/data/main_data.hdf5"), bottom, top)
        for agent_id in AGENT_LIST
    ]
    return [group_name, *[cell(value) for value in values]]


def save_csv(name: str, header: list[str], rows: list[list[str]]) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / name
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved: {out_path}")
    return out_path


def actual_rows() -> list[list[str]]:
    rows = []
    for (group_name, dataset_id), bottom_path, top_path in zip(DATASET_LIST, BOTTOM_LIST, TOP_LIST):
        rows.append(actual_row(group_name, bottom_path, dataset_id, top_path))
    return rows


def normalized_rows(*, use_max_top: bool) -> list[list[str]]:
    rows = []
    for (group_name, dataset_id), bottom_path, top_path in zip(DATASET_LIST, BOTTOM_LIST, TOP_LIST):
        rows.append(normalized_row(group_name, bottom_path, dataset_id, top_path, use_max_top=use_max_top))
    return rows


if __name__ == "__main__":
    save_csv(
        "actual_returns.csv",
        ["task", "random", *AGENT_LIST, "dataset"],
        actual_rows(),
    )
    save_csv(
        "mean_normalized_returns.csv",
        ["task", *AGENT_LIST],
        normalized_rows(use_max_top=False),
    )
    save_csv(
        "max_normalized_returns.csv",
        ["task", *AGENT_LIST],
        normalized_rows(use_max_top=True),
    )
