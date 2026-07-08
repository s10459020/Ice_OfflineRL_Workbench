from ice_offline.config.paths import _task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.run.eval import cal_eval
from ice_offline.run.plot import plot

DATASETS = [
    "hopper_d4rl_medium",
    "hopper_d4rl_hybrid",
    "hopper_d4rl_expert",
    "hopper_replay_medium",
    "hopper_replay_expert",
    "halfcheetah_d4rl_medium",
    "halfcheetah_d4rl_hybrid",
    "halfcheetah_d4rl_expert",
    "halfcheetah_replay_medium",
    "halfcheetah_replay_expert",
]

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl",
    "aspl_gp_punish_005",
    "aspl_gp_punish_010",
    "aspl_gp_punish_050",
    "aspl_r",
    "sdc",
    "sdc_cql",
    "scas",
    "scas_lambda_0",
    "scas_lambda_25",
    "scas_lambda_50",
    "scas_lambda_75",
    "scas_lambda_100",
    "scas_gp",
    "scaspl",
    "scaspl_gp",
]

MODELS = [
    "scas_model",
    "sdc_model",
]

def plot_item(dataset_id: str, id: str, eval_output_paths: list[str]) -> None:
    task_id = _task_id(dataset_id, id)
    metrics_output_path = metric_path(task_id)
    output_path = plot_path("train", dataset_id, id)

    print(f"plot dataset={dataset_id}, id={id}")
    plot([metrics_output_path], eval_output_paths, output_path)
    print(f"saved: {output_path}")


def plot_agent(dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    returns_output_path, steps_output_path = cal_eval(task_id, "train")
    plot_item(dataset_id, agent_id, [returns_output_path, steps_output_path])


def plot_model(dataset_id: str, model_id: str) -> None:
    plot_item(dataset_id, model_id, [])


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for agent_id in AGENTS:
            plot_agent(dataset_id, agent_id)
        for model_id in MODELS:
            plot_model(dataset_id, model_id)
