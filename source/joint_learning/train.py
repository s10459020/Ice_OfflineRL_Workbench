from joint_learning.agent import make_agent
from joint_learning.agent import model_path
from joint_learning.dataset import D4RLDataset


# Available agents: "bc", "td3bc", "iql", "cql"
AGENT = "td3bc"

# Available datasets:
# - "hopper_medium"
# - "hopper_expert"
# - "hopper_hybrid"
# - "hopper_medium_replay"
# - "hopper_expert_replay"
DATASET = "hopper_medium"

# Use only "cuda" or "cpu" to keep this build easy to reason about.
DEVICE = "cuda"
STEPS = 10_000
BATCH_SIZE = 256
PRINT_INTERVAL = 1_000


def main() -> None:
    dataset = D4RLDataset(DATASET, DEVICE)
    agent = make_agent(AGENT, dataset.obs_size, dataset.act_size, DEVICE)

    print(f"train agent={AGENT} dataset={DATASET} steps={STEPS} device={DEVICE}")
    for step in range(1, STEPS + 1):
        batch = dataset.sample_batch(BATCH_SIZE)
        agent.update(batch)

        if step % PRINT_INTERVAL == 0 or step == STEPS:
            print(f"step={step}")

    path = model_path(AGENT, DATASET)
    agent.save(path)
    print(f"saved model: {path}")


if __name__ == "__main__":
    main()
