from joint_learning.agents.aspl import ASPLAgent
from joint_learning.agents.aspl_c import ASPLCAgent
from joint_learning.agents.aspl_gp import ASPLGPAgent
from joint_learning.agents.bc import BCAgent
from joint_learning.agents.cql import CQLAgent
from joint_learning.agents.iql import IQLAgent
from joint_learning.agents.scas import SCASAgent
from joint_learning.agents.scas_gp import SCASGPAgent
from joint_learning.agents.scas_gpn import SCASGPNAgent
from joint_learning.agents.scas_n import SCASNAgent
from joint_learning.agents.scaspl import SCASPLAgent
from joint_learning.agents.scaspl_c import SCASPLCAgent
from joint_learning.agents.scaspl_gp import SCASPLGPAgent
from joint_learning.agents.scaspl_n import SCASPLNAgent
from joint_learning.agents.scaspl_nc import SCASPLNCAgent
from joint_learning.agents.scc import SCCAgent
from joint_learning.agents.scc_gp import SCCGPAgent
from joint_learning.agents.scc_gpn import SCCGPNAgent
from joint_learning.agents.scc_n import SCCNAgent
from joint_learning.agents.td3bc import TD3BCAgent
from joint_learning.agents.td3bc_gp import TD3BCGPAgent
from joint_learning.agents.td3bc_p import TD3BCPAgent
from joint_learning.agents.td3bc_p_gp import TD3BCPGPAgent
from joint_learning.agents.td3bc_xn import TD3BCXNAgent
from joint_learning.agents.td3bc_xn_gp import TD3BCXNGPAgent
from joint_learning.lib.train import train_model


# Available agents for this compact build:
# - "bc": behavior cloning baseline
# - "td3bc": original TD3BC with normalized actor objective
# - "td3bc_xn": TD3BC without actor objective normalization
# - "td3bc_p": TD3BC-XN with reduced actor objective weight
# - "td3bc_xn_gp": TD3BC-XN with action-gradient penalty
# - "td3bc_p_gp": TD3BC-P with action-gradient penalty
# - "td3bc_gp": original TD3BC with action-gradient penalty
# - "iql": implicit Q-learning baseline
# - "cql": conservative Q-learning baseline
# - "aspl": action-space pseudo-labeling
# - "aspl_c": ASPL with dataset-action compensation
# - "aspl_gp": ASPL with action-gradient penalty
# - "scas": state correction with action-space smoothing
# - "scas_n": SCAS with normalized actor objective
# - "scas_gp": SCAS with action-gradient penalty
# - "scas_gpn": SCAS-GP with normalized actor objective
# - "scaspl": SCAS with pseudo-label critic punishment
# - "scaspl_n": SCASPL with normalized actor objective
# - "scaspl_gp": SCASPL with action-gradient penalty
# - "scaspl_c": SCASPL with dataset-action compensation
# - "scaspl_nc": SCASPL-N with dataset-action compensation
# - "scc": SCAS with conservative critic
# - "scc_n": SCC with normalized actor objective
# - "scc_gp": SCC with action-gradient penalty
# - "scc_gpn": SCC-GP with normalized actor objective
AGENT_CLASSES = {
    "bc": BCAgent,
    "td3bc": TD3BCAgent,
    "td3bc_xn": TD3BCXNAgent,
    "td3bc_p": TD3BCPAgent,
    "td3bc_xn_gp": TD3BCXNGPAgent,
    "td3bc_p_gp": TD3BCPGPAgent,
    "td3bc_gp": TD3BCGPAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "aspl": ASPLAgent,
    "aspl_c": ASPLCAgent,
    "aspl_gp": ASPLGPAgent,
}

DYNAMIC_AGENT_CLASSES = {
    "scas": SCASAgent,
    "scas_n": SCASNAgent,
    "scas_gp": SCASGPAgent,
    "scas_gpn": SCASGPNAgent,
    "scaspl": SCASPLAgent,
    "scaspl_n": SCASPLNAgent,
    "scaspl_gp": SCASPLGPAgent,
    "scaspl_c": SCASPLCAgent,
    "scaspl_nc": SCASPLNCAgent,
    "scc": SCCAgent,
    "scc_n": SCCNAgent,
    "scc_gp": SCCGPAgent,
    "scc_gpn": SCCGPNAgent,
}


def make_agent(agent_id: str, dataset, device: str):
    if agent_id in DYNAMIC_AGENT_CLASSES:
        dynamics = train_model(dataset, device)
        agent_class = DYNAMIC_AGENT_CLASSES[agent_id]
        agent = agent_class(dataset.obs_size, dataset.act_size, dynamics=dynamics, device=device)
        agent.id = agent_id
        return agent

    agent_class = AGENT_CLASSES[agent_id]
    agent = agent_class(dataset.obs_size, dataset.act_size, device=device)
    agent.id = agent_id
    return agent
