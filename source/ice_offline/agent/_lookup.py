from collections.abc import Callable

from ice_offline.agent._spec import Agent
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.aspl_c import AsplCAgent
from ice_offline.agent.aspl_gp import AsplGPAgent
from ice_offline.agent.aspl_r import AsplRAgent
from ice_offline.agent.bc import BCAgent
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.cql_gp import CQLGPAgent
from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.agent.scas_adject import ScasAdjectAgent
from ice_offline.agent.scas_gp import ScasGPAgent
from ice_offline.agent.scas_gpn import ScasGPNAgent
from ice_offline.agent.scas_n import ScasNAgent
from ice_offline.agent.scc import SccAgent
from ice_offline.agent.scc_gpn import SccGPNAgent
from ice_offline.agent.scc_gp import SccGPAgent
from ice_offline.agent.scc_ns import SccNSAgent
from ice_offline.agent.scc_n import SccNAgent
from ice_offline.agent.iql import IQLAgent
from ice_offline.agent.scaspl import ScasplAgent
from ice_offline.agent.scaspl_c import ScasplCAgent
from ice_offline.agent.scaspl_gp import ScasplGPAgent
from ice_offline.agent.scaspl_gpn import ScasplGPNAgent
from ice_offline.agent.scaspl_nc import ScasplNCAgent
from ice_offline.agent.scaspl_ns import ScasplNSAgent
from ice_offline.agent.scaspl_n import ScasplNAgent
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3_gp import TD3GPAgent
from ice_offline.agent.td3_gpn import TD3GPNAgent
from ice_offline.agent.td3_n import TD3NAgent
from ice_offline.agent.td3_r import TD3RAgent
from ice_offline.agent.td3_s import TD3SAgent
from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.agent.td3bc_b import TD3BCBAgent
from ice_offline.agent.td3bc_bgp import TD3BCBGPAgent
from ice_offline.agent.td3bc_gp import TD3BCGPAgent
from ice_offline.agent.td3bc_gpn import TD3BCGPNAgent
from ice_offline.agent.td3bc_n import TD3BCNAgent
from ice_offline.agent.td3bc_r import TD3BCRAgent
from ice_offline.config.paths import model_path
from ice_offline.config.paths import task_id
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
    "td3_s": _agent(TD3SAgent),
    "td3_gamma_90": _agent(TD3Agent, discount_factor=0.9),
    "td3_n": _agent(TD3NAgent),
    "td3_r": _agent(TD3RAgent),
    "td3_gp": _agent(TD3GPAgent),
    "td3_gpn": _agent(TD3GPNAgent),
    "td3bc": _agent(TD3BCAgent),
    "td3bc_b": _agent(TD3BCBAgent),
    "td3bc_bgp": _agent(TD3BCBGPAgent),
    "td3bc_n": _agent(TD3BCNAgent),
    "td3bc_n_1": _agent(TD3BCNAgent, weight_td3=1.0),
    "td3bc_r": _agent(TD3BCRAgent),
    "td3bc_gp": _agent(TD3BCGPAgent),
    "td3bc_gpn": _agent(TD3BCGPNAgent),
    "iql": _agent(IQLAgent),
    "cql": _agent(CQLAgent),
    "cql_threshold_n5": _agent(CQLAgent, threshold=-5.0),
    "cql_threshold_5": _agent(CQLAgent, threshold=5.0),
    "cql_gp": _agent(CQLGPAgent),
    "aspl": _agent(AsplAgent),
    "aspl_c": _agent(AsplCAgent),
    "aspl_c_00005": _agent(AsplCAgent, weight_compensate=0.0005),
    "aspl_c_0001": _agent(AsplCAgent, weight_compensate=0.001),
    "aspl_c_0005": _agent(AsplCAgent, weight_compensate=0.005),
    "aspl_c_0050": _agent(AsplCAgent, weight_compensate=0.05),
    "aspl_c_0500": _agent(AsplCAgent, weight_compensate=0.5),
    "aspl_c_5000": _agent(AsplCAgent, weight_compensate=5.0),
    "aspl_gp_punish_005": _agent(AsplGPAgent, weight_punish=0.05),
    "aspl_gp_punish_010": _agent(AsplGPAgent, weight_punish=0.1),
    "aspl_gp_punish_050": _agent(AsplGPAgent, weight_punish=0.5),
    "aspl_r": _agent(AsplRAgent),
    "aspl_gamma_90": _agent(AsplAgent, discount_factor=0.9),
    "aspl_gamma_95": _agent(AsplAgent, discount_factor=0.95),
    "aspl_gp": _agent(AsplGPAgent),
}

MODEL_AGENT_TABLE: dict[str, Callable[..., Agent]] = {
    "scas": _model_agent(ScasAgent),
    "scas_lambda_0": _model_agent(ScasAgent, weight_correction=0.0),
    "scas_lambda_25": _model_agent(ScasAgent, weight_correction=0.25),
    "scas_lambda_50": _model_agent(ScasAgent, weight_correction=0.5),
    "scas_lambda_75": _model_agent(ScasAgent, weight_correction=0.75),
    "scas_lambda_100": _model_agent(ScasAgent, weight_correction=1.0),
    "scas_adject": _model_agent(ScasAdjectAgent),
    "scas_adject_01": _model_agent(ScasAdjectAgent, lambda_td3=0.1),
    "scas_adject_00075_00025": _model_agent(ScasAdjectAgent, lambda_td3=0.0075, lambda_corr=0.0025),
    "scas_adject_075_025": _model_agent(ScasAdjectAgent, lambda_td3=0.75, lambda_corr=0.25),
    "scas_adject_75_25": _model_agent(ScasAdjectAgent, lambda_td3=75.0, lambda_corr=25.0),
    "scas_adject_1": _model_agent(ScasAdjectAgent, lambda_td3=1.0),
    "scas_adject_1_01": _model_agent(ScasAdjectAgent, lambda_td3=1.0, lambda_corr=0.1),
    "scas_adject_5_5": _model_agent(ScasAdjectAgent, lambda_td3=5.0, lambda_corr=5.0),
    "scas_adject_10": _model_agent(ScasAdjectAgent, lambda_td3=10.0),
    "scc": _model_agent(SccAgent),
    "scc_ns": _model_agent(SccNSAgent),
    "scc_n": _model_agent(SccNAgent),
    "scc_gp": _model_agent(SccGPAgent),
    "scc_gpn": _model_agent(SccGPNAgent),
    "scc_gp_lambda_0": _model_agent(SccGPAgent, weight_correction=0.0),
    "scc_gp_lambda_100": _model_agent(SccGPAgent, weight_correction=1.0),
    "scas_n": _model_agent(ScasNAgent),
    "scas_n_lambda_0": _model_agent(ScasNAgent, weight_correction=0.0),
    "scas_n_lambda_100": _model_agent(ScasNAgent, weight_correction=1.0),
    "scas_gp": _model_agent(ScasGPAgent),
    "scas_gpn": _model_agent(ScasGPNAgent),
    "scaspl": _model_agent(ScasplAgent),
    "scaspl_c": _model_agent(ScasplCAgent),
    "scaspl_nc": _model_agent(ScasplNCAgent),
    "scaspl_n": _model_agent(ScasplNAgent),
    "scaspl_n_lambda_0": _model_agent(ScasplNAgent, weight_correction=0.0),
    "scaspl_n_lambda_100": _model_agent(ScasplNAgent, weight_correction=1.0),
    "scaspl_ns": _model_agent(ScasplNSAgent),
    "scaspl_gp": _model_agent(ScasplGPAgent),
    "scaspl_gpn": _model_agent(ScasplGPNAgent),
}

MODEL_AGENT_MODEL_TABLE: dict[str, str] = {
    "scas": "scas_model",
    "scas_lambda_0": "scas_model",
    "scas_lambda_25": "scas_model",
    "scas_lambda_50": "scas_model",
    "scas_lambda_75": "scas_model",
    "scas_lambda_100": "scas_model",
    "scas_adject": "scas_model",
    "scas_adject_01": "scas_model",
    "scas_adject_00075_00025": "scas_model",
    "scas_adject_075_025": "scas_model",
    "scas_adject_75_25": "scas_model",
    "scas_adject_1": "scas_model",
    "scas_adject_1_01": "scas_model",
    "scas_adject_5_5": "scas_model",
    "scas_adject_10": "scas_model",
    "scc": "scas_model",
    "scc_ns": "scas_model",
    "scc_n": "scas_model",
    "scc_gp": "scas_model",
    "scc_gpn": "scas_model",
    "scc_gp_lambda_0": "scas_model",
    "scc_gp_lambda_100": "scas_model",
    "scas_n": "scas_model",
    "scas_n_lambda_0": "scas_model",
    "scas_n_lambda_100": "scas_model",
    "scas_gp": "scas_model",
    "scas_gpn": "scas_model",
    "scaspl": "scas_model",
    "scaspl_c": "scas_model",
    "scaspl_nc": "scas_model",
    "scaspl_n": "scas_model",
    "scaspl_n_lambda_0": "scas_model",
    "scaspl_n_lambda_100": "scas_model",
    "scaspl_ns": "scas_model",
    "scaspl_gp": "scas_model",
    "scaspl_gpn": "scas_model",
    "sdc": "sdc_model",
    "sdc_cql": "sdc_model",
}


def make_model(id: str, dataset: Dataset, device: str = "cuda", **kwargs) -> Agent:
    config = dict(kwargs)
    model = MODEL_TABLE[id](dataset, device, config)
    model.id = id
    return model


def _require_model(
    id: str,
    dataset: Dataset,
    device: str,
    step: int | None,
    train_id: str | None,
) -> Agent:
    model = make_model(id, dataset, device)
    if step is not None:
        model_train_id = task_id(dataset.id, id)
        if train_id is not None:
            model_train_id = train_id
        path = model_path(model_train_id, step)
        if not path.exists():
            raise FileNotFoundError(f"missing model checkpoint: {path}; train {model_train_id} first")
        model.load(path)
    return model


def make_agent(
    id: str,
    dataset: Dataset,
    device: str = "cuda",
    model_step: int | None = None,
    model_train_id: str | None = None,
    **kwargs,
) -> Agent:
    config = dict(kwargs)
    if id in MODEL_AGENT_TABLE:
        model_id = MODEL_AGENT_MODEL_TABLE[id]
        model = _require_model(model_id, dataset, device, step=model_step, train_id=model_train_id)
        agent = MODEL_AGENT_TABLE[id](dataset, device, model, config)
    else:
        agent = AGENT_TABLE[id](dataset, device, config)
    agent.id = id
    return agent
