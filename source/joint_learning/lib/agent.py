from joint_learning.agents.aspl import ASPLAgent, ASPLC10Agent, ASPLCAgent, ASPLGPAgent
from joint_learning.agents.bc import BCAgent
from joint_learning.agents.cql import CQLAgent
from joint_learning.agents.iql import IQLAgent
from joint_learning.agents.scas import SCASAgent, SCASGPAgent, SCASGPNAgent, SCASNAgent
from joint_learning.agents.scaspl import SCASPLAgent, SCASPLCAgent, SCASPLGPAgent, SCASPLNAgent, SCASPLNCAgent
from joint_learning.agents.sccc import SCCCAgent, SCCCGPAgent, SCCCGPNAgent, SCCCNAgent
from joint_learning.agents.td3bc import TD3BCAgent, TD3BCGPAgent, TD3BCPGPAgent, TD3BCPAgent, TD3BCXNAgent, TD3BCXNGPAgent


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
# - "aspl_c_10": ASPL with dataset-action compensation weight 10
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
# - "sccc": SCAS with conservative critic
# - "sccc_n": SCCC with normalized actor objective
# - "sccc_gp": SCCC with action-gradient penalty
# - "sccc_gpn": SCCC-GP with normalized actor objective
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
    "aspl_c_10": ASPLC10Agent,
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
    "sccc": SCCCAgent,
    "sccc_n": SCCCNAgent,
    "sccc_gp": SCCCGPAgent,
    "sccc_gpn": SCCCGPNAgent,
}


def make_agent(agent_id: str, dataset, dynamic=None):
    if agent_id in DYNAMIC_AGENT_CLASSES:
        agent_class = DYNAMIC_AGENT_CLASSES[agent_id]
        agent = agent_class(dataset.obs_size, dataset.act_size, dynamic=dynamic, device=dataset.device)
        agent.id = agent_id
        return agent

    agent_class = AGENT_CLASSES[agent_id]
    agent = agent_class(dataset.obs_size, dataset.act_size, device=dataset.device)
    agent.id = agent_id
    return agent
