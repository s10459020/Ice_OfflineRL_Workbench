from ice_offline.agent.bc_continuous_deterministic import BCAgentContinuousDeterministic
from ice_offline.agent.bc_continuous_deterministic import eval_bc_deterministic_loss_pi
from ice_offline.agent.bc_continuous_stochastic import BCAgentContinuousStochastic
from ice_offline.agent.bc_continuous_stochastic import eval_bc_stochastic_loss_pi
from ice_offline.agent.bc_discrete import BCAgentDiscrete
from ice_offline.agent.bc_discrete import eval_bc_discrete_loss
from ice_offline.agent.cql_continuous import CQLAgentContinuous
from ice_offline.agent.cql_continuous import eval_cql_continuous_loss_pi
from ice_offline.agent.cql_continuous import eval_cql_continuous_loss_q
from ice_offline.agent.cql_discrete import CQLAgentDiscrete
from ice_offline.agent.cql_discrete import eval_cql_discrete_loss
from ice_offline.agent.cql_discrete import eval_cql_discrete_loss_cql
from ice_offline.agent.cql_discrete import eval_cql_discrete_loss_td
from ice_offline.agent.iql_continuous import IQLAgentContinuous
from ice_offline.agent.iql_continuous import eval_iql_continuous_loss_pi
from ice_offline.agent.iql_continuous import eval_iql_continuous_loss_q
from ice_offline.agent.iql_continuous import eval_iql_continuous_loss_v
from ice_offline.agent.iql_discrete import IQLAgentDiscrete
from ice_offline.agent.iql_discrete import eval_iql_discrete_loss
from ice_offline.agent.iql_discrete import eval_iql_discrete_loss_q
from ice_offline.agent.iql_discrete import eval_iql_discrete_loss_v
from ice_offline.agent.q_discrete import QAgentDiscrete
from ice_offline.agent.q_discrete import eval_q_discrete_loss
from ice_offline.agent.q_discrete import eval_q_discrete_loss_q
from ice_offline.agent.qv_discrete import QVAgentDiscrete
from ice_offline.agent.qv_discrete import eval_qv_discrete_loss
from ice_offline.agent.qv_discrete import eval_qv_discrete_loss_q
from ice_offline.agent.qv_discrete import eval_qv_discrete_loss_v
from ice_offline.run.evaluator import OfflineEvalFn
from ice_offline.run.offline import RunnerAgent


AGENT_LOOKUP: dict[str, RunnerAgent] = {
    "bc_deterministic": BCAgentContinuousDeterministic(),
    "bc_discrete": BCAgentDiscrete(),
    "bc_stochastic": BCAgentContinuousStochastic(),
    "cql_continuous": CQLAgentContinuous(),
    "cql_discrete": CQLAgentDiscrete(),
    "iql_continuous": IQLAgentContinuous(),
    "iql_discrete": IQLAgentDiscrete(),
    "q_discrete": QAgentDiscrete(),
    "qv_discrete": QVAgentDiscrete(),
}

AGENT_EVAL_OFFLINE_LOOKUP: dict[str, list[OfflineEvalFn]] = {
    "bc_deterministic": [eval_bc_deterministic_loss_pi],
    "bc_discrete": [eval_bc_discrete_loss],
    "bc_stochastic": [eval_bc_stochastic_loss_pi],
    "cql_continuous": [eval_cql_continuous_loss_q, eval_cql_continuous_loss_pi],
    "cql_discrete": [eval_cql_discrete_loss, eval_cql_discrete_loss_td, eval_cql_discrete_loss_cql],
    "iql_continuous": [eval_iql_continuous_loss_q, eval_iql_continuous_loss_v, eval_iql_continuous_loss_pi],
    "iql_discrete": [eval_iql_discrete_loss, eval_iql_discrete_loss_q, eval_iql_discrete_loss_v],
    "q_discrete": [eval_q_discrete_loss, eval_q_discrete_loss_q],
    "qv_discrete": [eval_qv_discrete_loss, eval_qv_discrete_loss_q, eval_qv_discrete_loss_v],
}


def get_agent(agent_id: str) -> RunnerAgent:
    return AGENT_LOOKUP[agent_id]


def get_agent_train_bundle(agent_id: str) -> tuple[RunnerAgent, list[OfflineEvalFn]]:
    return get_agent(agent_id), AGENT_EVAL_OFFLINE_LOOKUP[agent_id]

