from ice_offline.agent._spec import Agent
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.agent.cql_soft_q import CQLSoftQAgent
from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.halfcheetah_expert import HalfCheetahExpertDataset
from ice_offline.dataset.halfcheetah_expert_d4rl import HalfCheetahExpertD4rlDataset
from ice_offline.dataset.halfcheetah_medium import HalfCheetahMediumDataset
from ice_offline.dataset.halfcheetah_medium_d4rl import HalfCheetahMediumD4rlDataset
from ice_offline.dataset.halfcheetah_medium_expert import HalfCheetahMediumExpertDataset
from ice_offline.dataset.halfcheetah_medium_replay import HalfCheetahMediumReplayDataset
from ice_offline.dataset.halfcheetah_random import HalfCheetahRandomDataset
from ice_offline.dataset.halfcheetah_replay import HalfCheetahReplayDataset
from ice_offline.dataset.halfcheetah_simple import HalfCheetahSimpleDataset
from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.hopper_expert_d4rl import HopperExpertD4rlDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.dataset.hopper_random import HopperRandomDataset
from ice_offline.dataset.hopper_replay import HopperReplayDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.dataset.walker2d_expert import Walker2dExpertDataset
from ice_offline.dataset.walker2d_expert_d4rl import Walker2dExpertD4rlDataset
from ice_offline.dataset.walker2d_medium import Walker2dMediumDataset
from ice_offline.dataset.walker2d_medium_d4rl import Walker2dMediumD4rlDataset
from ice_offline.dataset.walker2d_medium_expert import Walker2dMediumExpertDataset
from ice_offline.dataset.walker2d_medium_replay import Walker2dMediumReplayDataset
from ice_offline.dataset.walker2d_random import Walker2dRandomDataset
from ice_offline.dataset.walker2d_replay import Walker2dReplayDataset
from ice_offline.dataset.walker2d_simple import Walker2dSimpleDataset


DEVICE = "cuda"


def make_dataset(id: str) -> Dataset:
    dataset_config: dict[str, Dataset] = {
        "hopper_random": HopperRandomDataset(device=DEVICE),
        "hopper_replay": HopperReplayDataset(device=DEVICE),
        "hopper_medium_replay": HopperMediumReplayDataset(device=DEVICE),
        "hopper_medium_d4rl": HopperMediumD4rlDataset(device=DEVICE),
        "hopper_medium_expert": HopperMediumExpertDataset(device=DEVICE),
        "hopper_expert_d4rl": HopperExpertD4rlDataset(device=DEVICE),
        "hopper_simple": HopperSimpleDataset(device=DEVICE),
        "hopper_medium": HopperMediumDataset(device=DEVICE),
        "hopper_expert": HopperExpertDataset(device=DEVICE),
        "walker2d_random": Walker2dRandomDataset(device=DEVICE),
        "walker2d_replay": Walker2dReplayDataset(device=DEVICE),
        "walker2d_medium_replay": Walker2dMediumReplayDataset(device=DEVICE),
        "walker2d_medium_d4rl": Walker2dMediumD4rlDataset(device=DEVICE),
        "walker2d_medium_expert": Walker2dMediumExpertDataset(device=DEVICE),
        "walker2d_expert_d4rl": Walker2dExpertD4rlDataset(device=DEVICE),
        "walker2d_simple": Walker2dSimpleDataset(device=DEVICE),
        "walker2d_medium": Walker2dMediumDataset(device=DEVICE),
        "walker2d_expert": Walker2dExpertDataset(device=DEVICE),
        "halfcheetah_random": HalfCheetahRandomDataset(device=DEVICE),
        "halfcheetah_replay": HalfCheetahReplayDataset(device=DEVICE),
        "halfcheetah_medium_replay": HalfCheetahMediumReplayDataset(device=DEVICE),
        "halfcheetah_medium_d4rl": HalfCheetahMediumD4rlDataset(device=DEVICE),
        "halfcheetah_medium_expert": HalfCheetahMediumExpertDataset(device=DEVICE),
        "halfcheetah_expert_d4rl": HalfCheetahExpertD4rlDataset(device=DEVICE),
        "halfcheetah_simple": HalfCheetahSimpleDataset(device=DEVICE),
        "halfcheetah_medium": HalfCheetahMediumDataset(device=DEVICE),
        "halfcheetah_expert": HalfCheetahExpertDataset(device=DEVICE),
    }
    return dataset_config[id]


def make_agent(id: str, dataset: Dataset) -> Agent:
    agent_config: dict[str, Agent] = {
        "bc_deterministic": BCDeterministicAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=DEVICE),
        "td3bc": TD3BCAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=DEVICE),
        "cql_soft_q": CQLSoftQAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=DEVICE),
        "aspl": AsplAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim, device=DEVICE),
    }
    return agent_config[id]
