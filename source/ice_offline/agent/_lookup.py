from collections.abc import Callable
from pathlib import Path

from ice_offline.agent._spec import Agent
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.agent.bc_stochastic import BCStochasticAgent
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.cql_max_q import CQLMaxQAgent
from ice_offline.agent.cql_soft_q import CQLSoftQAgent
from ice_offline.agent.iql import IQLAgent
from ice_offline.agent.scas_aspl import ScasAsplAgent
from ice_offline.agent.scas_min import ScasDynamic
from ice_offline.agent.scas_mean import ScasMeanAgent
from ice_offline.agent.scas_min import ScasMinAgent
from ice_offline.agent.sdc_cql import SDCCQLAgent
from ice_offline.agent.sdc_pre import SDCPreModel
from ice_offline.agent.sdc_pre import SDCPreAgent
from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.config.paths import model_path
from ice_offline.dataset.base import Dataset


DEFAULT_MODEL_STEP = 100_000


MODEL_TABLE: dict[str, Callable[[Dataset, str], Agent]] = {
    "scas_dynamic": lambda dataset, device: ScasDynamic(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device),
    "sdc_pre_model": lambda dataset, device: SDCPreModel(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device),
}


AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "bc_deterministic": lambda dataset, device, **agent_kwargs: BCDeterministicAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "bc_stochastic": lambda dataset, device, **agent_kwargs: BCStochasticAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "td3bc": lambda dataset, device, **agent_kwargs: TD3BCAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "iql": lambda dataset, device, **agent_kwargs: IQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "cql": lambda dataset, device, **agent_kwargs: CQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "cql_max_q": lambda dataset, device, **agent_kwargs: CQLMaxQAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "cql_soft_q": lambda dataset, device, **agent_kwargs: CQLSoftQAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "aspl": lambda dataset, device, **agent_kwargs: AsplAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
    "sdc_cql": lambda dataset, device, **agent_kwargs: SDCCQLAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=device, **agent_kwargs),
}


def make_model(id: str, dataset: Dataset, device: str = "cuda") -> Agent:
    model = MODEL_TABLE[id](dataset, device)
    model.id = id
    return model


def _require_model(id: str, dataset: Dataset, device: str, step: int = DEFAULT_MODEL_STEP) -> Agent:
    model = make_model(id, dataset, device)
    path = model_path(dataset.id, step).with_suffix(".pt")
    if not path.exists():
        raise FileNotFoundError(f"missing model checkpoint: {path}; train {id} first")
    model.load(dataset.id, step)
    return model


def make_agent(id: str, dataset: Dataset, device: str = "cuda", **agent_kwargs) -> Agent:
    if id == "sdc_pre":
        state_models = _require_model("sdc_pre_model", dataset, device)
        agent = SDCPreAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, state_models=state_models, device=device)
    elif id == "scas_min":
        dynamics = _require_model("scas_dynamic", dataset, device)
        agent = ScasMinAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, dynamics=dynamics, device=device)
    elif id == "scas_mean":
        dynamics = _require_model("scas_dynamic", dataset, device)
        agent = ScasMeanAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, dynamics=dynamics, device=device)
    elif id == "scas_aspl":
        dynamics = _require_model("scas_dynamic", dataset, device)
        agent = ScasAsplAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, dynamics=dynamics, device=device)
    else:
        agent = AGENT_TABLE[id](dataset, device, **agent_kwargs)
    agent.id = id
    return agent
