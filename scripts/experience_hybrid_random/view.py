from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import main_data_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import table_path
from ice_offline.run.boxplot import boxplot
from ice_offline.run.eval import cal_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true

TABLES = [
    ("walker2d_random_expert_9", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_7", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_5", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_3", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_1", "walker2d_random", "walker2d_d4rl_expert"),
]

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    # "scaspl",
]


def _ensure_agent_eval(dataset_id: str, agent_id: str) -> object:
    task_id = _task_id(dataset_id, agent_id)
    input_path = main_data_path("test", task_id)
    if not input_path.exists():
        print(f"skip missing: {input_path}")
        return None

    returns_output_path, _ = cal_main(task_id)
    print(f"saved: {returns_output_path}")
    return returns_output_path


def _ensure_dataset_eval(dataset_id: str) -> object:
    output_path = returns_path("dataset", dataset_id)
    if output_path.exists():
        return output_path
    returns_output_path, _ = cal_dataset(dataset_id)
    print(f"saved: {returns_output_path}")
    return returns_output_path


def _ordered_table_dataset_ids(table_specs_list: list[tuple[str, str, str]]) -> list[str]:
    lowers = [lower_id for _, lower_id, _ in table_specs_list]
    datasets = [dataset_id for dataset_id, _, _ in table_specs_list]
    uppers = [upper_id for _, _, upper_id in table_specs_list]
    dataset_ids: list[str] = []
    for id in [*lowers, *datasets, *uppers]:
        if id not in dataset_ids:
            dataset_ids.append(id)
    return dataset_ids


def ensure_table_datasets(table_specs_list: list[tuple[str, str, str]]) -> None:
    for dataset_id in _ordered_table_dataset_ids(table_specs_list):
        _ensure_dataset_eval(dataset_id)


def save_tables(dataset_id_list: list[str], agent_id_list: list[str]) -> None:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids = [dataset_id for dataset_id, _, _ in table_specs_list]
    datas: list[list[object]] = []
    lowers: list[object] = []
    uppers: list[object] = []

    for dataset_id, lower_id, upper_id in table_specs_list:
        datas.append([
            returns_path("test", _task_id(dataset_id, agent_id))
            for agent_id in agent_id_list
        ])
        lowers.append(returns_path("dataset", lower_id))
        uppers.append(returns_path("dataset", upper_id))

    table_true(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_hybrid_random", "true_returns.csv"))
    table_mean(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_hybrid_random", "mean_returns.csv"))
    table_pr95(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_hybrid_random", "pr95_returns.csv"))


def save_table_boxplot(table_specs_list: list[tuple[str, str, str]]) -> None:
    table_members = [
        (dataset_id, returns_path("dataset", dataset_id))
        for dataset_id in _ordered_table_dataset_ids(table_specs_list)
    ]

    table_output_path = VIEW_ROOT / "boxplot" / "experience_hybrid_random" / "table.png"
    table_path = boxplot("table", table_members, table_output_path)
    if table_path is not None:
        print(f"saved: {table_path}")


def save_boxplots(dataset_id_list: list[str], agent_id_list: list[str]) -> None:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    for dataset_id, lower_id, upper_id in table_specs_list:
        members = [("lower", returns_path("dataset", lower_id))]
        for agent_id in agent_id_list:
            members.append((agent_id, returns_path("test", _task_id(dataset_id, agent_id))))
        members.append(("upper", returns_path("dataset", upper_id)))

        output_path = VIEW_ROOT / "boxplot" / "experience_hybrid_random" / f"{dataset_id}.png"
        path = boxplot(dataset_id, members, output_path)
        if path is not None:
            print(f"saved: {path}")


if __name__ == "__main__":
    ensure_table_datasets(TABLES)
    for dataset_id, _, _ in TABLES:
        for agent_id in AGENTS:
            _ensure_agent_eval(dataset_id, agent_id)

    save_tables(DATASETS, AGENTS)
    save_table_boxplot(TABLES)
    save_boxplots(DATASETS, AGENTS)
