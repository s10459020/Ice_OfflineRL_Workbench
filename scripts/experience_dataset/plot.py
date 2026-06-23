from ice_offline.config.paths import _task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.run.eval import cal_eval
from ice_offline.run.plot import plot


AGENT_TASKS = [
    ("hopper_d4rl_medium", "bc_deterministic"),
    ("hopper_d4rl_hybrid", "bc_deterministic"),
    ("hopper_d4rl_expert", "bc_deterministic"),
    ("hopper_replay_medium", "bc_deterministic"),
    ("hopper_replay_expert", "bc_deterministic"),
    ("hopper_d4rl_medium", "bc_stochastic"),
    ("hopper_d4rl_hybrid", "bc_stochastic"),
    ("hopper_d4rl_expert", "bc_stochastic"),
    ("hopper_replay_medium", "bc_stochastic"),
    ("hopper_replay_expert", "bc_stochastic"),
    ("hopper_d4rl_medium", "td3bc"),
    ("hopper_d4rl_hybrid", "td3bc"),
    ("hopper_d4rl_expert", "td3bc"),
    ("hopper_replay_medium", "td3bc"),
    ("hopper_replay_expert", "td3bc"),
    ("hopper_d4rl_medium", "iql"),
    ("hopper_d4rl_hybrid", "iql"),
    ("hopper_d4rl_expert", "iql"),
    ("hopper_replay_medium", "iql"),
    ("hopper_replay_expert", "iql"),
    ("hopper_d4rl_medium", "cql"),
    ("hopper_d4rl_hybrid", "cql"),
    ("hopper_d4rl_expert", "cql"),
    ("hopper_replay_medium", "cql"),
    ("hopper_replay_expert", "cql"),
    ("hopper_d4rl_medium", "aspl"),
    ("hopper_d4rl_hybrid", "aspl"),
    ("hopper_d4rl_expert", "aspl"),
    ("hopper_replay_medium", "aspl"),
    ("hopper_replay_expert", "aspl"),
    ("hopper_d4rl_medium", "sdc"),
    ("hopper_d4rl_hybrid", "sdc"),
    ("hopper_d4rl_expert", "sdc"),
    ("hopper_replay_medium", "sdc"),
    ("hopper_replay_expert", "sdc"),
    ("hopper_d4rl_medium", "sdc_cql"),
    ("hopper_d4rl_hybrid", "sdc_cql"),
    ("hopper_d4rl_expert", "sdc_cql"),
    ("hopper_replay_medium", "sdc_cql"),
    ("hopper_replay_expert", "sdc_cql"),
    ("hopper_d4rl_medium", "scas"),
    ("hopper_d4rl_hybrid", "scas"),
    ("hopper_d4rl_expert", "scas"),
    ("hopper_replay_medium", "scas"),
    ("hopper_replay_expert", "scas"),
    ("hopper_d4rl_medium", "scas_aspl"),
    ("hopper_d4rl_hybrid", "scas_aspl"),
    ("hopper_d4rl_expert", "scas_aspl"),
    ("hopper_replay_medium", "scas_aspl"),
    ("hopper_replay_expert", "scas_aspl"),
]

MODEL_TASKS = [
    ("hopper_d4rl_medium", "scas_model"),
    ("hopper_d4rl_hybrid", "scas_model"),
    ("hopper_d4rl_expert", "scas_model"),
    ("hopper_random", "scas_model"),
    ("hopper_replay_medium", "scas_model"),
    ("hopper_replay_expert", "scas_model"),
    ("hopper_d4rl_medium", "sdc_model"),
    ("hopper_d4rl_hybrid", "sdc_model"),
    ("hopper_d4rl_expert", "sdc_model"),
    ("hopper_random", "sdc_model"),
    ("hopper_replay_medium", "sdc_model"),
    ("hopper_replay_expert", "sdc_model"),
]

def plot_item(index: int, dataset_id: str, id: str, eval_output_paths: list[str]) -> None:
    task_id = _task_id(dataset_id, id)
    metrics_output_path = metric_path(task_id)
    output_path = plot_path(index, dataset_id, id)

    print(f"plot dataset={dataset_id}, id={id}")
    plot([metrics_output_path], eval_output_paths, output_path)
    print(f"saved: {output_path}")


def plot_agent(index: int, dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    returns_output_path, steps_output_path = cal_eval(task_id, "train")
    plot_item(index, dataset_id, agent_id, [returns_output_path, steps_output_path])


def plot_model(index: int, dataset_id: str, model_id: str) -> None:
    plot_item(index, dataset_id, model_id, [])


if __name__ == "__main__":
    dataset_ids = list(dict.fromkeys(dataset_id for dataset_id, _ in AGENT_TASKS + MODEL_TASKS))
    for dataset_id, agent_id in AGENT_TASKS:
        plot_agent(dataset_ids.index(dataset_id) + 1, dataset_id, agent_id)
    for dataset_id, model_id in MODEL_TASKS:
        plot_model(dataset_ids.index(dataset_id) + 1, dataset_id, model_id)
