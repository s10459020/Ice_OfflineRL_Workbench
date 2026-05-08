from typing import Any, Callable

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
from ice_offline.runner.offline import BatchType
from ice_offline.runner.offline import OfflineEvalFn
from ice_offline.runner.offline import RunnerAgent
from ice_offline.runner.offline import TransitionBatch


AgentBuilder = Callable[[int, int], RunnerAgent]


class LazyAgentProxy:
    def __init__(self, builder: AgentBuilder):
        self.device = "cpu"
        self._builder = builder
        self._inner: RunnerAgent | None = None

    def set_dim(self, obs_size: int, act_size: int) -> None:
        self._inner = self._builder(obs_size, act_size)
        self.device = self._inner.device

    def act_best(self, observation: Any) -> Any:
        return self._inner.act_best(observation)

    def update(self, batch: BatchType) -> None:
        self._inner.update(batch)

    def save(self, model_name) -> Any:
        return self._inner.save(model_name)

    def load(self, model_name) -> None:
        self._inner.load(model_name)

    def __getattr__(self, item: str):
        return getattr(self._inner, item)


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


def eval_bc_deterministic_loss_pi(
    agent: ContinuousBCDeterministicAgent,
    transitions: TransitionBatch,
) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_pi": float(agent.loss_actor(o, a).item())}


def eval_bc_stochastic_loss_pi(
    agent: ContinuousBCStochasticAgent,
    transitions: TransitionBatch,
) -> dict[str, float]:
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


AGENT_BUILDERS: dict[str, AgentBuilder] = {
    "bc_discrete": lambda obs_size, act_size: DiscreteBCAgent(obs_size=obs_size, act_size=act_size),
    "q_discrete": lambda obs_size, act_size: QAgentDiscrete(obs_size=obs_size, act_size=act_size),
    "qv_discrete": lambda obs_size, act_size: QVAgentDiscrete(obs_size=obs_size, act_size=act_size),
    "iql_discrete": lambda obs_size, act_size: IQLAgentDiscrete(obs_size=obs_size, act_size=act_size),
    "cql_discrete": lambda obs_size, act_size: CQLAgentDiscrete(obs_size=obs_size, act_size=act_size),
    "bc_deterministic": lambda obs_size, act_size: ContinuousBCDeterministicAgent(obs_size=obs_size, act_size=act_size),
    "bc_stochastic": lambda obs_size, act_size: ContinuousBCStochasticAgent(obs_size=obs_size, act_size=act_size),
    "iql_continuous": lambda obs_size, act_size: IQLAgentContinuous(obs_size=obs_size, act_size=act_size),
    "cql_continuous": lambda obs_size, act_size: CQLAgentContinuous(obs_size=obs_size, act_size=act_size),
}


AGENT_EVAL_OFFLINE_LOOKUP: dict[str, list[OfflineEvalFn]] = {
    "bc_discrete": [eval_bc_discrete_loss],
    "q_discrete": [eval_q_discrete_loss, eval_q_discrete_loss_q],
    "qv_discrete": [eval_qv_discrete_loss, eval_qv_discrete_loss_q, eval_qv_discrete_loss_v],
    "iql_discrete": [eval_iql_discrete_loss, eval_iql_discrete_loss_q, eval_iql_discrete_loss_v],
    "cql_discrete": [eval_cql_discrete_loss, eval_cql_discrete_loss_td, eval_cql_discrete_loss_cql],
    "bc_deterministic": [eval_bc_deterministic_loss_pi],
    "bc_stochastic": [eval_bc_stochastic_loss_pi],
    "iql_continuous": [eval_iql_continuous_loss_q, eval_iql_continuous_loss_v, eval_iql_continuous_loss_pi],
    "cql_continuous": [eval_cql_continuous_loss_q, eval_cql_continuous_loss_pi],
}


def get_agent(agent_id: str) -> RunnerAgent:
    return LazyAgentProxy(AGENT_BUILDERS[agent_id])


def get_agent_train_bundle(agent_id: str) -> tuple[RunnerAgent, list[OfflineEvalFn]]:
    agent = get_agent(agent_id)
    eval_offline_fns = AGENT_EVAL_OFFLINE_LOOKUP[agent_id]
    return agent, eval_offline_fns
