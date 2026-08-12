from joint_learning.lib.agent import make_agent
from joint_learning.lib.dataset import D4RLDataset
from joint_learning.lib.table import table_path
from joint_learning.lib.table import write_table
from joint_learning.lib.test import test
from joint_learning.lib.train import train


EXPERIMENT = "benchmark"
DEVICE = "cuda"

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl_c",
    "scas_n",
    "scaspl_n",
    "scc_n",
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


def main() -> None:
    rows: list[list[str]] = []
    header = ["dataset"] + AGENTS

    for dataset_id in DATASETS:
        row = [dataset_id]
        for agent_id in AGENTS:
            dataset = D4RLDataset(dataset_id, DEVICE)
            agent = make_agent(agent_id, dataset, DEVICE)
            train(agent, dataset, EXPERIMENT)
            result = test(agent, dataset)
            row.append(f"{result.mean_return:.6f}")
        rows.append(row)
        write_table(table_path(EXPERIMENT), header, rows)


if __name__ == "__main__":
    main()
