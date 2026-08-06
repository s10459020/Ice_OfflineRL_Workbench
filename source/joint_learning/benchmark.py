from joint_learning.agents.lib import make_agent
from joint_learning.datasets.lib import D4RLDataset
from joint_learning.utils.train import train


DEVICE = "cuda"

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
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
    for agent_id in AGENTS:
        for dataset_id in DATASETS:
            dataset = D4RLDataset(dataset_id, DEVICE)
            agent = make_agent(agent_id, dataset.obs_size, dataset.act_size, DEVICE)
            train(agent, dataset, agent_id)


if __name__ == "__main__":
    main()
