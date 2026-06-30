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
    ("noise_state_5e-3@hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("noise_state_5e-3@hopper_d4rl_expert", "hopper_random", "hopper_d4rl_expert"),
    ("noise_state_5e-3@hopper_replay_medium", "hopper_random", "hopper_replay_medium"),
    ("noise_state_5e-3@hopper_replay_expert", "hopper_random", "hopper_replay_expert"),
]

DATASETS = [
    "noise_state_5e-3@hopper_d4rl_medium",
    "noise_state_5e-3@hopper_d4rl_expert",
    "noise_state_5e-3@hopper_replay_medium",
    "noise_state_5e-3@hopper_replay_expert",
]

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl",
    "scas",
    "scaspl",
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

    table_true(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_noise_state", "true_returns.csv"))
    table_mean(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_noise_state", "mean_returns.csv"))
    table_pr95(dataset_ids, agent_id_list, datas, lowers, uppers, table_path("experience_noise_state", "pr95_returns.csv"))


def save_boxplots(dataset_id_list: list[str], agent_id_list: list[str]) -> None:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    for index, (dataset_id, lower_id, upper_id) in enumerate(table_specs_list, start=1):
        members = [("lower", returns_path("dataset", lower_id))]
        for agent_id in agent_id_list:
            members.append((agent_id, returns_path("test", _task_id(dataset_id, agent_id))))
        members.append(("upper", returns_path("dataset", upper_id)))

        output_path = VIEW_ROOT / "boxplot" / "experience_noise_state" / f"{index}. {dataset_id}.png"
        path = boxplot(dataset_id, members, output_path)
        if path is not None:
            print(f"saved: {path}")


if __name__ == "__main__":
    for dataset_id, lower_id, upper_id in TABLES:
        _ensure_dataset_eval(lower_id)
        _ensure_dataset_eval(upper_id)
        for agent_id in AGENTS:
            _ensure_agent_eval(dataset_id, agent_id)

    save_tables(DATASETS, AGENTS)
    save_boxplots(DATASETS, AGENTS)
