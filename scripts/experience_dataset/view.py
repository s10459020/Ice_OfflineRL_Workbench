from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.config.paths import main_data_path
from ice_offline.config.paths import plot_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.config.paths import table_path
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.boxplot import boxplot
from ice_offline.run.eval import cal_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.eval import eval_returns
from ice_offline.run.eval import eval_steps
from ice_offline.run.eval import read_eval
from ice_offline.run.eval import write_eval_rows
from ice_offline.run.plot import plot_overlay
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true

TABLES = [
    ("hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("hopper_d4rl_expert", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_replay_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_replay_expert", "hopper_random", "hopper_d4rl_expert"),
    ("walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_hybrid"),
    ("walker2d_d4rl_expert", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("walker2d_replay_expert", "walker2d_random", "walker2d_d4rl_expert"),
    # ("halfcheetah_d4rl_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    # ("halfcheetah_d4rl_hybrid", "halfcheetah_random", "halfcheetah_d4rl_hybrid"),
    # ("halfcheetah_d4rl_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
    # ("halfcheetah_replay_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    # ("halfcheetah_replay_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
]

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
]

AGENTS = [
    # "bc",
    # "td3bc",
    # "iql",
    # "cql",
    # "aspl",
    # "aspl_r",
    # "scas",
    # "scas_gp",
    # "scaspl",
    # "scaspl_gp",
]


def _plot_agent_eval(task_id: str) -> object:
    eval_data = read_eval("test", task_id)
    returns_rows = eval_data["returns"][1]
    dataset_id, agent_id, _ = task_id.rsplit("-", 2)
    series_list = [
        (str(step), list(range(1, len(values) + 1)), values)
        for step, values in returns_rows
    ]
    output_path = plot_path("dataset", dataset_id, agent_id)
    path = plot_overlay(task_id, series_list, output_path)
    print(f"saved: {path}")
    return path


def ensure_agent_eval(dataset_id: str, agent_id: str) -> object:
    task_id = _task_id(dataset_id, agent_id)
    eval_path = eval_data_path("test", task_id)
    if eval_path.exists():
        returns_output_path = returns_path("test", task_id)
        steps_output_path = steps_path("test", task_id)
        if not returns_output_path.exists() or not steps_output_path.exists():
            eval_dataset = EvalDataset(path=eval_path, device="cpu")
            batches = eval_dataset.batch_episodes
            returns_rows = eval_returns(batches)
            steps_rows = eval_steps(batches)
            returns_output_path, _ = write_eval_rows("test", task_id, returns_rows, steps_rows)
            print(f"saved: {returns_output_path}")
        _plot_agent_eval(task_id)
        return returns_output_path

    input_path = main_data_path("test", task_id)
    if not input_path.exists():
        print(f"skip missing: {eval_path}")
        print(f"skip missing: {input_path}")
        return None

    returns_output_path, _ = cal_main(task_id)
    print(f"saved: {returns_output_path}")
    return returns_output_path


def ensure_dataset_eval(dataset_id: str) -> object:
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
        ensure_dataset_eval(dataset_id)


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

    table_true(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_dataset", "true_returns.csv"))
    table_mean(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_dataset", "mean_returns.csv"))
    table_pr95(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_dataset", "pr95_returns.csv"))


def save_table_boxplot(table_specs_list: list[tuple[str, str, str]]) -> None:
    table_members = [
        (dataset_id, returns_path("dataset", dataset_id))
        for dataset_id in _ordered_table_dataset_ids(table_specs_list)
    ]

    table_output_path = VIEW_ROOT / "boxplot" / "experience_dataset" / "table.png"
    path = boxplot("table", table_members, table_output_path)
    if path is not None:
        print(f"saved: {path}")


def save_boxplots(dataset_id_list: list[str], agent_id_list: list[str]) -> None:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    for dataset_id, lower_id, upper_id in table_specs_list:
        members = [("lower", returns_path("dataset", lower_id))]
        for agent_id in agent_id_list:
            members.append((agent_id, returns_path("test", _task_id(dataset_id, agent_id))))
        members.append(("upper", returns_path("dataset", upper_id)))

        output_path = VIEW_ROOT / "boxplot" / "experience_dataset" / f"{dataset_id}.png"
        path = boxplot(dataset_id, members, output_path)
        if path is not None:
            print(f"saved: {path}")


if __name__ == "__main__":
    for dataset_id, _, _ in TABLES:
        for agent_id in AGENTS:
            ensure_agent_eval(dataset_id, agent_id)

    ensure_table_datasets(TABLES)
    save_tables(DATASETS, AGENTS)
    save_table_boxplot(TABLES)
    save_boxplots(DATASETS, AGENTS)
