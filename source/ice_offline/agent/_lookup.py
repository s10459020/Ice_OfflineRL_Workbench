from collections.abc import Callable

from ice_offline.agent._spec import Agent
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.bc import BCAgent
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.agent.scas_gp import ScasGPAgent
from ice_offline.agent.iql import IQLAgent
from ice_offline.agent.scaspl import ScasplAgent
from ice_offline.agent.scaspl_gp import ScasplGPAgent
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3_gp import TD3GPAgent
from ice_offline.agent.td3_gpn import TD3GPNAgent
from ice_offline.agent.td3_n import TD3NAgent
from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.agent.td3bc_gp import TD3BCGPAgent
from ice_offline.agent.td3bc_gpn import TD3BCGPNAgent
from ice_offline.agent.td3bc_n import TD3BCNAgent
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
}

AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "bc": _agent(BCAgent),
    "td3": _agent(TD3Agent),
    "td3_n": _agent(TD3NAgent),
    "td3_gp": _agent(TD3GPAgent),
    "td3_gpn": _agent(TD3GPNAgent),
    "td3bc": _agent(TD3BCAgent),
    "td3bc_n": _agent(TD3BCNAgent),
    "td3bc_gp": _agent(TD3BCGPAgent),
    "td3bc_gpn": _agent(TD3BCGPNAgent),
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
    "scas_gp": _model_agent(ScasGPAgent),
    "scaspl": _model_agent(ScasplAgent),
    "scaspl_gp": _model_agent(ScasplGPAgent),
}

MODEL_AGENT_MODEL_TABLE: dict[str, str] = {
    "scas": "scas_model",
    "scas_lambda_0": "scas_model",
    "scas_lambda_25": "scas_model",
    "scas_lambda_50": "scas_model",
    "scas_lambda_75": "scas_model",
    "scas_lambda_100": "scas_model",
    "scas_gp": "scas_model",
    "scaspl": "scas_model",
    "scaspl_gp": "scas_model",
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
