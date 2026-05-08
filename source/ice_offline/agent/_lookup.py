import torch

from ice_offline.agent import CQLAgentContinuous
from ice_offline.agent import CQLAgentDiscrete
from ice_offline.agent import ContinuousBCDeterministicAgent
from ice_offline.agent import ContinuousBCStochasticAgent
from ice_offline.agent import DiscreteBCAgent
from ice_offline.agent import IQLAgentContinuous
from ice_offline.agent import IQLAgentDiscrete
from ice_offline.agent import QAgentDiscrete
from ice_offline.agent import QVAgentDiscrete
from ice_offline.runner.offline import OfflineEvalFn
from ice_offline.runner.offline import RunnerAgent
from ice_offline.runner.offline import TransitionBatch


def eval_bc_discrete_loss(agent: DiscreteBCAgent, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss": float(agent.loss_actor(o, a).item())}


def eval_q_discrete_loss(agent: QAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss": float(agent.loss_critic(o, a, r, on, d).item())}


def eval_q_discrete_loss_q(agent: QAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss_q": float(agent._loss_q(o, a, r, on, d).item())}


def eval_qv_discrete_loss(agent: QVAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss": float(agent.loss_critic(o, a, r, on, d).item())}


def eval_qv_discrete_loss_q(agent: QVAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss_q": float(agent._loss_q(o, a, r, on, d).item())}


def eval_qv_discrete_loss_v(agent: QVAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_v": float(agent._loss_v(o, a).item())}


def eval_iql_discrete_loss(agent: IQLAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss": float(agent.loss_critic(o, a, r, on, d).item())}


def eval_iql_discrete_loss_q(agent: IQLAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss_q": float(agent._loss_q(o, a, r, on, d).item())}


def eval_iql_discrete_loss_v(agent: IQLAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_v": float(agent._loss_v(o, a).item())}


def eval_cql_discrete_loss(agent: CQLAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss": float(agent.loss_critic(o, a, r, on, d).item())}


def eval_cql_discrete_loss_td(agent: CQLAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss_td": float(agent._loss_td(o, a, r, on, d).item())}


def eval_cql_discrete_loss_cql(agent: CQLAgentDiscrete, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_cql": float(agent._loss_cql(o, a).item())}


def eval_bc_deterministic_loss_pi(agent: ContinuousBCDeterministicAgent, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_pi": float(agent.loss_actor(o, a).item())}


def eval_bc_stochastic_loss_pi(agent: ContinuousBCStochasticAgent, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_pi": float(agent.loss_actor(o, a).item())}


def eval_iql_continuous_loss_q(agent: IQLAgentContinuous, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    with torch.no_grad():
        return {"loss_q": float(agent._loss_q(o, a, r, on, d).item())}


def eval_iql_continuous_loss_v(agent: IQLAgentContinuous, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    with torch.no_grad():
        return {"loss_v": float(agent._loss_v(o, a).item())}


def eval_iql_continuous_loss_pi(agent: IQLAgentContinuous, transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    with torch.no_grad():
        return {"loss_pi": float(agent.loss_actor(o, a).item())}


def eval_cql_continuous_loss_q(agent: CQLAgentContinuous, transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    with torch.no_grad():
        return {"loss_q": float(agent.loss_critic(o, a, r, on, d).item())}


def eval_cql_continuous_loss_pi(agent: CQLAgentContinuous, transitions: TransitionBatch) -> dict[str, float]:
    o, _, _, _, _ = transitions
    with torch.no_grad():
        return {"loss_pi": float(agent.loss_actor(o).item())}


AGENT_LOOKUP: dict[str, RunnerAgent] = {
    "bc_deterministic": ContinuousBCDeterministicAgent(),
    "bc_discrete": DiscreteBCAgent(),
    "bc_stochastic": ContinuousBCStochasticAgent(),
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
