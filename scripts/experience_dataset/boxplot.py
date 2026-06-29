from ice_offline.config.paths import VIEW_ROOT
from ice_offline.run.boxplot import boxplot
from ice_offline.run.eval import cal_dataset
from prepare import eval_agent_result

DATASETS = [
    ("hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("hopper_d4rl_expert", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_replay_medium", "hopper_random", "hopper_replay_medium"),
    ("hopper_replay_expert", "hopper_random", "hopper_replay_expert"),
]

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl",
    "scas",
    "scaspl",
]

def save_boxplot(index: int, dataset_id: str, lower_id: str, upper_id: str) -> None:
    lower_returns_path, _ = cal_dataset(lower_id)
    upper_returns_path, _ = cal_dataset(upper_id)
    members = [("random", lower_returns_path)]
    for agent_id in AGENTS:
        members.append((agent_id, eval_agent_result(dataset_id, agent_id)))
    members.append(("dataset", upper_returns_path))

    output_path = VIEW_ROOT / "boxplot" / "experience_dataset" / f"{index}. {dataset_id}.png"
    path = boxplot(dataset_id, members, output_path)
    if path is not None:
        print(f"saved: {path}")


def build_boxplots() -> None:
    for index, (dataset_id, lower_id, upper_id) in enumerate(DATASETS, start=1):
        save_boxplot(index, dataset_id, lower_id, upper_id)


if __name__ == "__main__":
    build_boxplots()
