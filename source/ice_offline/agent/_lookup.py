from ice_offline.agent.bc_continuous_deterministic import BCAgentContinuousDeterministic
from ice_offline.agent.bc_continuous_stochastic import BCAgentContinuousStochastic
from ice_offline.agent.bc_discrete import BCAgentDiscrete
from ice_offline.agent.cql_continuous import CQLAgentContinuous
from ice_offline.agent.cql_discrete import CQLAgentDiscrete
from ice_offline.agent.iql_continuous import IQLAgentContinuous
from ice_offline.agent.iql_discrete import IQLAgentDiscrete
from ice_offline.agent._spec import TorchAgent


AGENT_LOOKUP = {
    "bc_deterministic": BCAgentContinuousDeterministic,
    "bc_discrete": BCAgentDiscrete,
    "bc_stochastic": BCAgentContinuousStochastic,
    "cql_continuous": CQLAgentContinuous,
    "cql_discrete": CQLAgentDiscrete,
    "iql_continuous": IQLAgentContinuous,
    "iql_discrete": IQLAgentDiscrete,
}


def get_agent(agent_id: str, obs_size: int, act_size: int) -> TorchAgent:
    return AGENT_LOOKUP[agent_id](obs_size=obs_size, act_size=act_size)
