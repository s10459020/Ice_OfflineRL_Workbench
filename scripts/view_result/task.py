from pathlib import Path

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
from ice_offline.tools.paths import dataset_root

RANDOM_DATASET_CLASS_BY_ENV_NAME = {
    "hopper": HopperRandomDataset,
    "halfcheetah": HalfCheetahRandomDataset,
    "walker2d": Walker2dRandomDataset,
}


def dataset_id(dataset_cls) -> str:
    return dataset_cls().id


def dataset_group_name(dataset_cls) -> str:
    return dataset_view_name(dataset_cls)


def dataset_env_name(dataset_cls) -> str:
    return dataset_id(dataset_cls).split("_", 1)[0]


def dataset_view_name(dataset_cls) -> str:
    return dataset_id(dataset_cls).split("_", 1)[1]


def view_root(dataset_cls) -> Path:
    return Path("tmps/view") / dataset_env_name(dataset_cls)


def plot_output_path(index: int, dataset_cls, agent_id: str) -> Path:
    return view_root(dataset_cls) / "plot" / agent_id / f"{index}. {dataset_view_name(dataset_cls)}.png"


def boxplot_output_path(index: int, dataset_cls) -> Path:
    return view_root(dataset_cls) / "boxplot" / f"{index}. {dataset_view_name(dataset_cls)}.png"


def table_output_path(dataset_cls, name: str) -> Path:
    return view_root(dataset_cls) / "table" / name


def is_d4rl_path(path: str) -> bool:
    return path.startswith("tmps/datasets/d4rl/")


def source_path(dataset_cls) -> str:
    dataset = dataset_cls()
    path = Path(dataset.path)
    if is_d4rl_path(path.as_posix()):
        return path.as_posix()
    return path.relative_to(dataset_root()).as_posix()


def test_path(dataset_cls, agent_id: str) -> str:
    dataset = dataset_cls()
    return f"test/{dataset.id}-{agent_id}-v0/data/main_data.hdf5"


def bottom_path(dataset_cls) -> str:
    path = source_path(dataset_cls)
    if is_d4rl_path(path):
        random_dataset_cls = RANDOM_DATASET_CLASS_BY_ENV_NAME[dataset_env_name(dataset_cls)]
        return source_path(random_dataset_cls)
    return test_path(dataset_cls, "random")


def top_path(dataset_cls) -> str:
    if dataset_cls is RANDOM_DATASET_CLASS_BY_ENV_NAME[dataset_env_name(dataset_cls)]:
        return ""
    return source_path(dataset_cls)
