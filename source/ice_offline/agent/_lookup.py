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


DEFAULT_MODEL_STEP = 100_000


MODEL_TABLE: dict[str, Callable[..., Agent]] = {
    "scas_model": lambda dataset, device, **kwargs: ScasDynamic(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
    "sdc_model": lambda dataset, device, **kwargs: SDCModel(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
}

AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "bc_deterministic": lambda dataset, device, **kwargs: BCDeterministicAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
    "bc_stochastic": lambda dataset, device, **kwargs: BCStochasticAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
    "td3": lambda dataset, device, **kwargs: TD3Agent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
    "td3bc": lambda dataset, device, **kwargs: TD3BCAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
    "iql": lambda dataset, device, **kwargs: IQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
    "cql": lambda dataset, device, **kwargs: CQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
    "aspl": lambda dataset, device, **kwargs: AsplAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **kwargs),
}

MODEL_AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "scas": lambda dataset, device, model, **kwargs: ScasAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, dynamics=model, device=device, **kwargs),
    "scas_aspl": lambda dataset, device, model, **kwargs: ScasAsplAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, dynamics=model, device=device, **kwargs),
    "sdc": lambda dataset, device, model, **kwargs: SDCAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, model=model, device=device, **kwargs),
    "sdc_cql": lambda dataset, device, model, **kwargs: SDCCQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, model=model, device=device, **kwargs),
}

MODEL_AGENT_MODEL_TABLE: dict[str, str] = {
    "scas": "scas_model",
    "scas_aspl": "scas_model",
    "sdc": "sdc_model",
    "sdc_cql": "sdc_model",
}


def make_model(id: str, dataset: Dataset, device: str = "cuda", **kwargs) -> Agent:
    model = MODEL_TABLE[id](dataset, device, **kwargs)
    model.id = id
    return model


def _require_model(id: str, dataset: Dataset, device: str, step: int = DEFAULT_MODEL_STEP) -> Agent:
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
    if id in MODEL_AGENT_TABLE:
        model_id = MODEL_AGENT_MODEL_TABLE[id]
        step = DEFAULT_MODEL_STEP if model_step is None else model_step
        model = _require_model(model_id, dataset, device, step=step)
        agent = MODEL_AGENT_TABLE[id](dataset, device, model, **kwargs)
    else:
        agent = AGENT_TABLE[id](dataset, device, **kwargs)
    agent.id = id
    return agent
