from joint_learning.lib.agent import DYNAMIC_AGENT_CLASSES
from joint_learning.lib.agent import make_agent
from joint_learning.lib.dataset import D4RLDataset
from joint_learning.lib.table import table_path
from joint_learning.lib.table import write_table
from joint_learning.lib.train import train
from joint_learning.lib.train import train_dynamic
from joint_learning.lib.metrics import clear_metric
from joint_learning.lib.metrics import metrics_path


EXPERIMENT = "benchmark"
DEVICE = "cuda"
TRAIN_COUNT = 5

AGENTS = [
    "bc",
    # "td3bc",
    # "iql",
    # "cql",
    # "aspl_c",
    # "scas_n",
    # "scaspl_n",
    # "sccc_n",
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
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = train_dynamic(dataset) if any(agent_id in DYNAMIC_AGENT_CLASSES for agent_id in AGENTS) else None
        for agent_id in AGENTS:
            clear_metric(metrics_path(agent_id, dataset_id))
            returns = []
            for train_index in range(1, TRAIN_COUNT + 1):
                print(f"train {train_index}/{TRAIN_COUNT} agent={agent_id} dataset={dataset_id}")
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                returns.append(train(agent, dataset, EXPERIMENT, train_index))
            mean_return = sum(returns) / len(returns)
            row.append(f"{mean_return:.6f}")
        rows.append(row)
        write_table(table_path(EXPERIMENT), header, rows)


if __name__ == "__main__":
    main()
