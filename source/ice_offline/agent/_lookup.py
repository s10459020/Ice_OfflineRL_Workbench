from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.agent.bc_stochastic import BCStochasticAgent
from ice_offline.agent.discrete.bc_discrete import BCDiscreteAgent
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.cql_max_q import CQLMaxQAgent
from ice_offline.agent.cql_soft_q import CQLSoftQAgent
from ice_offline.agent.discrete.cql_discrete import CQLDiscreteAgent
from ice_offline.agent.iql import IQLAgent
from ice_offline.agent.discrete.iql_discrete import IQLDiscreteAgent
from ice_offline.agent.sdc_cql import SDCCQLAgent
from ice_offline.agent.sdc_pre import SDCPreAgent
from ice_offline.agent.scas_aspl import ScasAsplAgent
from ice_offline.agent._spec import Agent


AGENT_LOOKUP = {
    "bc_deterministic": BCDeterministicAgent,
    "bc_discrete": BCDiscreteAgent,
    "bc_stochastic": BCStochasticAgent,
    "cql": CQLAgent,
    "cql_max_q": CQLMaxQAgent,
    "cql_soft_q": CQLSoftQAgent,
    "cql_discrete": CQLDiscreteAgent,
    "iql": IQLAgent,
    "iql_discrete": IQLDiscreteAgent,
    "sdc_cql": SDCCQLAgent,
    "sdc_pre": SDCPreAgent,
    "scas_aspl": ScasAsplAgent,
}


def get_agent(agent_id: str, obs_size: int, act_size: int) -> Agent:
    return AGENT_LOOKUP[agent_id](obs_size=obs_size, act_size=act_size)
