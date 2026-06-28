from collections.abc import Callable

from ice_offline.agent._spec import Agent
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.agent.iql import IQLAgent
from ice_offline.agent.scaspl import ScasplAgent
from ice_offline.agent.sdc import SDCAgent
from ice_offline.agent.sdc import SDCModel
from ice_offline.agent.sdc_cql import SDCCQLAgent
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import model_path
from ice_offline.dataset.base import Dataset


def _agent(agent_cls: type[Agent], **fixed_config: object) -> Callable[..., Agent]:
    return lambda dataset, device, config: agent_cls(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        config=fixed_config | config,
        device=device,
    )


def _model_agent(agent_cls: type[Agent], **fixed_config: object) -> Callable[..., Agent]:
    return lambda dataset, device, model, config: agent_cls(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        dynamics=model,
        config=fixed_config | config,
        device=device,
    )


MODEL_TABLE: dict[str, Callable[..., Agent]] = {
    "scas_model": lambda dataset, device, config: ScasDynamic(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "sdc_model": lambda dataset, device, config: SDCModel(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
}

AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "bc": _agent(BCDeterministicAgent),
    "td3": _agent(TD3Agent),
    "td3_q2": _agent(TD3Agent, q_count=2),
    "td3_q4": _agent(TD3Agent, q_count=4),
    "td3_q8": _agent(TD3Agent, q_count=8),
    "td3bc": _agent(TD3BCAgent),
    "iql": _agent(IQLAgent),
    "cql": _agent(CQLAgent),
    "aspl": _agent(AsplAgent),
}

MODEL_AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "scas": _model_agent(ScasAgent),
    "scas_lambda_0": _model_agent(ScasAgent, weight_correction=0.0),
    "scas_lambda_25": _model_agent(ScasAgent, weight_correction=0.25),
    "scas_lambda_50": _model_agent(ScasAgent, weight_correction=0.5),
    "scas_lambda_75": _model_agent(ScasAgent, weight_correction=0.75),
    "scas_lambda_100": _model_agent(ScasAgent, weight_correction=1.0),
    "scaspl": _model_agent(ScasplAgent),
    "sdc": lambda dataset, device, model, config: SDCAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, model=model, config=config, device=device),
    "sdc_cql": lambda dataset, device, model, config: SDCCQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, model=model, config=config, device=device),
}

MODEL_AGENT_MODEL_TABLE: dict[str, str] = {
    "scas": "scas_model",
    "scas_lambda_0": "scas_model",
    "scas_lambda_25": "scas_model",
    "scas_lambda_50": "scas_model",
    "scas_lambda_75": "scas_model",
    "scas_lambda_100": "scas_model",
    "scaspl": "scas_model",
    "sdc": "sdc_model",
    "sdc_cql": "sdc_model",
}


def make_model(id: str, dataset: Dataset, device: str = "cuda", **kwargs) -> Agent:
    config = dict(kwargs)
    model = MODEL_TABLE[id](dataset, device, config)
    model.id = id
    return model


def _require_model(id: str, dataset: Dataset, device: str, step: int) -> Agent:
    model = make_model(id, dataset, device)
    task_id = _task_id(dataset.id, id)
    path = model_path(task_id, step).with_suffix(".pt")
    if not path.exists():
        raise FileNotFoundError(f"missing model checkpoint: {path}; train {id} first")
    model.load(task_id, step)
    return model


def make_agent(
    id: str,
    dataset: Dataset,
    device: str = "cuda",
    model_step: int | None = None,
    **kwargs,
) -> Agent:
    config = dict(kwargs)
    if id in MODEL_AGENT_TABLE:
        model_id = MODEL_AGENT_MODEL_TABLE[id]
        model = _require_model(model_id, dataset, device, step=model_step)
        agent = MODEL_AGENT_TABLE[id](dataset, device, model, config)
    else:
        agent = AGENT_TABLE[id](dataset, device, config)
    agent.id = id
    return agent
