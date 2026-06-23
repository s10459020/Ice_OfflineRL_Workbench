from collections.abc import Callable

from ice_offline.agent._spec import Agent
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.agent.bc_stochastic import BCStochasticAgent
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.agent.iql import IQLAgent
from ice_offline.agent.scas_aspl import ScasAsplAgent
from ice_offline.agent.sdc import SDCAgent
from ice_offline.agent.sdc import SDCModel
from ice_offline.agent.sdc_cql import SDCCQLAgent
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import model_path
from ice_offline.dataset.base import Dataset


MODEL_TABLE: dict[str, Callable[..., Agent]] = {
    "scas_model": lambda dataset, device, config: ScasDynamic(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "sdc_model": lambda dataset, device, config: SDCModel(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
}

AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "bc_deterministic": lambda dataset, device, config: BCDeterministicAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "bc_stochastic": lambda dataset, device, config: BCStochasticAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "td3": lambda dataset, device, config: TD3Agent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "td3bc": lambda dataset, device, config: TD3BCAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "iql": lambda dataset, device, config: IQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "cql": lambda dataset, device, config: CQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
    "aspl": lambda dataset, device, config: AsplAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, config=config, device=device),
}

MODEL_AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "scas": lambda dataset, device, model, config: ScasAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, dynamics=model, config=config, device=device),
    "scas_aspl": lambda dataset, device, model, config: ScasAsplAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, dynamics=model, config=config, device=device),
    "sdc": lambda dataset, device, model, config: SDCAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, model=model, config=config, device=device),
    "sdc_cql": lambda dataset, device, model, config: SDCCQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, model=model, config=config, device=device),
}

MODEL_AGENT_MODEL_TABLE: dict[str, str] = {
    "scas": "scas_model",
    "scas_aspl": "scas_model",
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
