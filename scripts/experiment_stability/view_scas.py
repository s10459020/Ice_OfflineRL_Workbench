from pathlib import Path

from ice_offline.run.table import write_tables
from view import DATASETS
from view import TABLES
from view import _agent_value
from view import _dataset_value

GROUP = "stability/scas"

AGENTS = [
    "scas",
    "scas_n",
    "scas_gp",
    "scas_gpn",
]


def save_tables_group(group: str, dataset_id_list: list[str], agent_id_list: list[str]) -> tuple[Path, Path, Path, Path]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))

    data_values = [
        [_agent_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_dataset_value(lower_id) for lower_id in lower_ids]
    upper_values = [_dataset_value(upper_id) for upper_id in upper_ids]

    return write_tables(
        group,
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


if __name__ == "__main__":
    save_tables_group(GROUP, DATASETS, AGENTS)
