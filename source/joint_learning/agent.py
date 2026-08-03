from pathlib import Path

from joint_learning.agents.bc import BCAgent
from joint_learning.agents.cql import CQLAgent
from joint_learning.agents.iql import IQLAgent
from joint_learning.agents.td3bc import TD3BCAgent


# Available agents for this compact build:
# - "bc": behavior cloning baseline
# - "td3bc": TD3 with behavior cloning regularization
# - "iql": implicit Q-learning baseline
# - "cql": conservative Q-learning baseline
AGENT_CLASSES = {
    "bc": BCAgent,
    "td3bc": TD3BCAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
}


def make_agent(agent_id: str, obs_size: int, act_size: int, device: str):
    agent_class = AGENT_CLASSES[agent_id]
    return agent_class(obs_size, act_size, device)


def model_path(agent_id: str, dataset_id: str) -> Path:
    return Path(__file__).resolve().parent / "model" / f"{agent_id}-{dataset_id}.pt"
