import json
from dataclasses import dataclass
from pathlib import Path

from ice_offline.config.paths import RUNS_ROOT
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
from ice_offline.dataset.hopper_expert_one import HopperExpertOneDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_one import HopperMediumOneDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.dataset.hopper_random import HopperRandomDataset
from ice_offline.dataset.hopper_replay import HopperReplayDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.dataset.hopper_simple_one import HopperSimpleOneDataset
from ice_offline.dataset.walker2d_expert import Walker2dExpertDataset
from ice_offline.dataset.walker2d_expert_d4rl import Walker2dExpertD4rlDataset
from ice_offline.dataset.walker2d_medium import Walker2dMediumDataset
from ice_offline.dataset.walker2d_medium_d4rl import Walker2dMediumD4rlDataset
from ice_offline.dataset.walker2d_medium_expert import Walker2dMediumExpertDataset
from ice_offline.dataset.walker2d_medium_replay import Walker2dMediumReplayDataset
from ice_offline.dataset.walker2d_random import Walker2dRandomDataset
from ice_offline.dataset.walker2d_replay import Walker2dReplayDataset
from ice_offline.dataset.walker2d_simple import Walker2dSimpleDataset


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    env_id: str
    path: Path
    dataset_cls: type


SOURCE_DATASET_CLASSES = [
    HopperRandomDataset,
    HopperReplayDataset,
    HopperMediumReplayDataset,
    HopperMediumD4rlDataset,
    HopperMediumExpertDataset,
    HopperExpertD4rlDataset,
    HopperSimpleDataset,
    HopperMediumDataset,
    HopperExpertDataset,
    HopperSimpleOneDataset,
    HopperMediumOneDataset,
    HopperExpertOneDataset,
    HalfCheetahRandomDataset,
    HalfCheetahReplayDataset,
    HalfCheetahMediumReplayDataset,
    HalfCheetahMediumD4rlDataset,
    HalfCheetahMediumExpertDataset,
    HalfCheetahExpertD4rlDataset,
    HalfCheetahSimpleDataset,
    HalfCheetahMediumDataset,
    HalfCheetahExpertDataset,
    Walker2dRandomDataset,
    Walker2dReplayDataset,
    Walker2dMediumReplayDataset,
    Walker2dMediumD4rlDataset,
    Walker2dMediumExpertDataset,
    Walker2dExpertD4rlDataset,
    Walker2dSimpleDataset,
    Walker2dMediumDataset,
    Walker2dExpertDataset,
]


def source_dataset_entries() -> list[DatasetEntry]:
    return [
        DatasetEntry(
            id=dataset_cls.id,
            env_id=dataset_cls.env_id,
            path=Path(dataset_cls.path),
            dataset_cls=dataset_cls,
        )
        for dataset_cls in SOURCE_DATASET_CLASSES
    ]


DATASET_CLASS_BY_ID = {
    dataset_cls.id: dataset_cls
    for dataset_cls in SOURCE_DATASET_CLASSES
}


def run_dataset_entries() -> list[DatasetEntry]:
    entries = []
    for path in sorted(RUNS_ROOT.glob("*/*/data/main_data.hdf5")):
        metadata = _read_metadata(path.with_name("metadata.json"))
        entries.append(
            DatasetEntry(
                id=metadata.get("id", ""),
                env_id=metadata.get("env_id", ""),
                path=path,
                dataset_cls=DATASET_CLASS_BY_ID[metadata.get("id", "")],
            )
        )
    return entries


def dataset_entries() -> list[DatasetEntry]:
    return source_dataset_entries() + run_dataset_entries()


def _read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
