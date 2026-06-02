from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from _return import returns


PLOT_ROOT = Path("plot")


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

D4RL_RANDOM_RETURNS = "tmps/datasets/d4rl/hopper_random-v2.hdf5"


def agent_member(dataset_id: str, agent_id: str) -> tuple[str, str]:
    return agent_id, f"test/{dataset_id}-{agent_id}-v0/data/main_data.hdf5"


def agent_members(dataset_id: str) -> list[tuple[str, str]]:
    return [agent_member(dataset_id, agent_id) for agent_id in AGENT_LIST]


GROUPS = [
    (
        "random",
        [
            ("random_dataset", D4RL_RANDOM_RETURNS),
            *agent_members("hopper_random"),
        ],
    ),
    (
        "replay",
        [
            ("random_dataset", D4RL_RANDOM_RETURNS),
            *agent_members("hopper_replay"),
            ("dataset", "tmps/datasets/d4rl/hopper_full_replay-v2.hdf5"),
        ],
    ),
    (
        "expert_d4rl",
        [
            ("random_dataset", D4RL_RANDOM_RETURNS),
            *agent_members("hopper_expert_d4rl"),
            ("dataset", "tmps/datasets/d4rl/hopper_expert-v2.hdf5"),
        ],
    ),
    (
        "medium_d4rl",
        [
            ("random_dataset", D4RL_RANDOM_RETURNS),
            *agent_members("hopper_medium_d4rl"),
            ("dataset", "tmps/datasets/d4rl/hopper_medium-v2.hdf5"),
        ],
    ),
    (
        "medium_replay",
        [
            ("random_dataset", D4RL_RANDOM_RETURNS),
            *agent_members("hopper_medium_replay"),
            ("dataset", "tmps/datasets/d4rl/hopper_medium_replay-v2.hdf5"),
        ],
    ),
    (
        "medium_expert",
        [
            ("random_dataset", D4RL_RANDOM_RETURNS),
            *agent_members("hopper_medium_expert"),
            ("dataset", "tmps/datasets/d4rl/hopper_medium_expert-v2.hdf5"),
        ],
    ),
    (
        "simple",
        [
            agent_member("hopper_simple", "random"),
            *agent_members("hopper_simple"),
            ("dataset", "mujoco/hopper/simple-v0/data/main_data.hdf5"),
        ],
    ),
    (
        "medium",
        [
            agent_member("hopper_medium", "random"),
            *agent_members("hopper_medium"),
            ("dataset", "mujoco/hopper/medium-v0/data/main_data.hdf5"),
        ],
    ),
    (
        "expert",
        [
            agent_member("hopper_expert", "random"),
            *agent_members("hopper_expert"),
            ("dataset", "mujoco/hopper/expert-v0/data/main_data.hdf5"),
        ],
    ),
]


def save_boxplot(index: int, group_name: str, members: list[tuple[str, str]]) -> Path:
    labels = [label for label, _ in members]
    values = [returns(dataset_path) for _, dataset_path in members]

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_ROOT / f"{index}. {group_name}.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_title(group_name)
    ax.set_ylabel("Return")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    for i, (group_name, members) in enumerate(GROUPS, start=1):
        print(f"group={group_name}")
        save_boxplot(i, group_name, members)
