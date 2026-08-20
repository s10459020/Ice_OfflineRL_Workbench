from joint_learning.lib.agent import DYNAMIC_AGENT_CLASSES
from joint_learning.lib.agent import make_agent
from joint_learning.lib.dataset import D4RLDataset
from joint_learning.lib.table import table_path
from joint_learning.lib.table import write_table
from joint_learning.lib.train import train
from joint_learning.lib.train import train_dynamic
from joint_learning.lib.metrics import clear_metric
from joint_learning.lib.metrics import metrics_path


EXPERIMENT = "stability_td3bc"
DEVICE = "cuda"
N_MODEL = 5
N_TRAIN = 200_000
N_EVAL = 10
BATCH_SIZE = 256
PRINT_INTERVAL = 2_000
RETURN_AVG_WINDOW = 10
MODEL_STEPS = 500_000
MODEL_BATCH_SIZE = 256
MODEL_PRINT_INTERVAL = 10_000

AGENTS = [
    "td3bc",
]

DATASETS = [
    "hopper_random",
    "hopper_medium",
    "hopper_medium_replay",
    "hopper_hybrid",
    "hopper_expert",
    "walker2d_random",
    "walker2d_medium",
    "walker2d_medium_replay",
    "walker2d_hybrid",
    "walker2d_expert",
    "halfcheetah_random",
    "halfcheetah_medium",
    "halfcheetah_medium_replay",
    "halfcheetah_hybrid",
    "halfcheetah_expert",
]


def main() -> None:
    rows: list[list[str]] = []
    header = ["dataset"] + AGENTS

    for dataset_id in DATASETS:
        row = [dataset_id]
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = (
            train_dynamic(dataset, MODEL_STEPS, MODEL_BATCH_SIZE, MODEL_PRINT_INTERVAL)
            if any(agent_id in DYNAMIC_AGENT_CLASSES for agent_id in AGENTS)
            else None
        )
        for agent_id in AGENTS:
            clear_metric(metrics_path(agent_id, dataset_id))
            returns = []
            for train_index in range(1, N_MODEL + 1):
                print(
                    f"train {train_index}/{N_MODEL} "
                    f"agent={agent_id} dataset={dataset_id}"
                )
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                returns.append(
                    train(
                        agent,
                        dataset,
                        EXPERIMENT,
                        train_index,
                        N_TRAIN,
                        N_EVAL,
                        BATCH_SIZE,
                        PRINT_INTERVAL,
                        RETURN_AVG_WINDOW,
                    )
                )
            mean_return = sum(returns) / len(returns)
            row.append(f"{mean_return:.6f}")
        rows.append(row)
        write_table(table_path(EXPERIMENT), header, rows)


if __name__ == "__main__":
    main()
