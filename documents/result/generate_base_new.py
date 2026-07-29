from pathlib import Path

from generate_result_quick_tables import BASE_TABLES
from generate_result_quick_tables import REPRESENTATIVE_AGENTS
from generate_result_quick_tables import ExperimentSpec
from generate_result_quick_tables import select_cell
from generate_result_quick_tables import write_experiment_table


BASE_NEW_SPEC = ExperimentSpec(
    "base_new",
    "base",
    "base_train",
    None,
    BASE_TABLES,
    REPRESENTATIVE_AGENTS,
)


def generate() -> Path:
    cells = [
        select_cell(BASE_NEW_SPEC, dataset, agent)
        for dataset in BASE_NEW_SPEC.datasets
        for agent in BASE_NEW_SPEC.agents
    ]
    return write_experiment_table(BASE_NEW_SPEC, cells)


if __name__ == "__main__":
    path = generate()
    print(f"saved: {path}")
