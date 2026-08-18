import csv
from pathlib import Path

from joint_learning.lib.dataset import D4RLDataset
from joint_learning.lib.metrics import load_metric
from joint_learning.lib.metrics import metrics_path


PROJECT_ROOT = Path(__file__).resolve().parent
TABLE_ROOT = PROJECT_ROOT / "table"
TRUE_PATH = TABLE_ROOT / "benchmark_table_true.csv"
PERCENT_PATH = TABLE_ROOT / "benchmark_table_percent.csv"
DEVICE = "cuda"

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl_c",
    "scas_n",
    "scaspl_n",
    "sccc_n",
]

DATASETS = [
    "hopper_medium",
    "hopper_expert",
    "hopper_hybrid",
    "hopper_medium_replay",
    "hopper_expert_replay",
    "walker2d_medium",
    "walker2d_expert",
    "walker2d_hybrid",
    "walker2d_medium_replay",
    "walker2d_expert_replay",
    "halfcheetah_medium",
    "halfcheetah_expert",
    "halfcheetah_hybrid",
    "halfcheetah_medium_replay",
    "halfcheetah_expert_replay",
]

REFERENCE_KINDS = {"medium", "expert", "hybrid"}
DATASET_SUFFIXES = (
    "_medium_replay",
    "_expert_replay",
    "_medium",
    "_expert",
    "_hybrid",
)


def environment_id(dataset_id: str) -> str:
    for suffix in DATASET_SUFFIXES:
        if dataset_id.endswith(suffix):
            return dataset_id[: -len(suffix)]
    return dataset_id


def dataset_kind(dataset_id: str) -> str:
    environment = environment_id(dataset_id)
    return dataset_id[len(environment) + 1 :]


def dataset_return(dataset_id: str) -> float:
    dataset = D4RLDataset(dataset_id, "cpu")
    rewards = dataset.rewards.detach().cpu().numpy().reshape(-1)
    dones = dataset.dones.detach().cpu().numpy().reshape(-1).astype(bool)
    episode_returns = []
    episode_return = 0.0
    for reward, done in zip(rewards, dones):
        episode_return += float(reward)
        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
    if episode_return != 0.0:
        episode_returns.append(episode_return)
    return sum(episode_returns) / len(episode_returns)


def true_rows() -> list[list[str]]:
    rows = []
    for dataset_id in DATASETS:
        row = [dataset_id]
        for agent_id in AGENTS:
            path = metrics_path(agent_id, dataset_id)
            if not path.exists():
                print(f"skip agent={agent_id} dataset={dataset_id} reason=metric_missing")
                row.append("")
                continue
            row.append(f"{load_metric(path):.6f}")
        rows.append(row)
        print(f"task complete dataset={dataset_id}")
    return rows


def percent_rows(rows: list[list[str]]) -> list[list[str]]:
    upper_returns: dict[tuple[str, str], float] = {}
    for dataset_id in DATASETS:
        kind = dataset_kind(dataset_id)
        if kind not in REFERENCE_KINDS:
            continue
        try:
            upper_returns[(environment_id(dataset_id), kind)] = dataset_return(dataset_id)
        except FileNotFoundError:
            continue

    result = []
    for row in rows:
        environment = environment_id(row[0])
        kind = dataset_kind(row[0])
        upper_kind = kind.removesuffix("_replay")
        values = [row[0]]
        for index, agent_id in enumerate(AGENTS, start=1):
            if row[index] == "":
                values.append("")
                continue
            bound = upper_returns.get((environment, upper_kind))
            if bound is None:
                values.append("")
                continue
            if bound == 0.0:
                values.append("")
                continue
            values.append(f"{float(row[index]) / bound * 100.0:.2f}")
        result.append(values)
    return result


def write_table(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", *AGENTS])
        writer.writerows(rows)


def main() -> None:
    rows = true_rows()
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    write_table(TRUE_PATH, rows)
    write_table(PERCENT_PATH, percent_rows(rows))


if __name__ == "__main__":
    main()
