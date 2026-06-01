from ice_offline.agent.bc_deterministic import BCAgentDeterministic
from ice_offline.agent.bc_stochastic import BCAgentStochastic
from ice_offline.agent.discrete.bc_discrete import BCAgentDiscrete
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.cql_max_q import CQLAgentMaxQ
from ice_offline.agent.cql_soft_q import CQLAgentSoftQ
from ice_offline.agent.discrete.cql_discrete import CQLAgentDiscrete
from ice_offline.agent.iql import IQLAgent
from ice_offline.agent.discrete.iql_discrete import IQLAgentDiscrete
from ice_offline.agent._spec import TorchAgent


AGENT_LOOKUP = {
    "bc_deterministic": BCAgentDeterministic,
    "bc_discrete": BCAgentDiscrete,
    "bc_stochastic": BCAgentStochastic,
    "cql": CQLAgent,
    "cql_max_q": CQLAgentMaxQ,
    "cql_soft_q": CQLAgentSoftQ,
    "cql_discrete": CQLAgentDiscrete,
    "iql": IQLAgent,
    "iql_discrete": IQLAgentDiscrete,
}


def get_agent(agent_id: str, obs_size: int, act_size: int) -> TorchAgent:
    return AGENT_LOOKUP[agent_id](obs_size=obs_size, act_size=act_size)
