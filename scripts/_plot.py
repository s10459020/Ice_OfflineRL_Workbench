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
from view_returns import plot


DATASET_CLASS_LIST = [
    HopperRandomDataset,
    HopperReplayDataset,
    HopperMediumReplayDataset,
    HopperMediumD4rlDataset,
    HopperMediumExpertDataset,
    HopperExpertD4rlDataset,
    HopperSimpleDataset,
    HopperMediumDataset,
    HopperExpertDataset,
    # Walker2dRandomDataset,
    # Walker2dReplayDataset,
    # Walker2dMediumReplayDataset,
    # Walker2dMediumD4rlDataset,
    # Walker2dMediumExpertDataset,
    # Walker2dExpertD4rlDataset,
    # Walker2dSimpleDataset,
    # Walker2dMediumDataset,
    # Walker2dExpertDataset,
    # HalfCheetahRandomDataset,
    # HalfCheetahReplayDataset,
    # HalfCheetahMediumReplayDataset,
    # HalfCheetahMediumD4rlDataset,
    # HalfCheetahMediumExpertDataset,
    # HalfCheetahExpertD4rlDataset,
    # HalfCheetahSimpleDataset,
    # HalfCheetahMediumDataset,
    # HalfCheetahExpertDataset,
]

AGENT_ID_LIST = [
    # "random",
    # "bc_deterministic",
    # "bc_stochastic",
    # "td3bc",
    # "iql",
    # "cql",
    # "cql_max_q",
    "cql_soft_q",
    # "aspl",
    # "sdc_cql",
    # "sdc_pre",
    # "scas_min",
    # "scas_mean",
    # "scas_aspl",
]


def main() -> None:
    plot.main(DATASET_CLASS_LIST, AGENT_ID_LIST)


if __name__ == "__main__":
    main()
