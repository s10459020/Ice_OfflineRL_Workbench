from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.run.eval import cal_returns
from ice_offline.run.eval import cal_steps
from ice_offline.run.plot import plot
from ice_offline.run.train import train


TRAIN_KWARGS = {
    # "start": 500_000,
    "steps": 200_000,
    # "save_interval": 20_000,
    # "eval_interval": 2_000,
    # "print_interval": 200,
    # "eval_episodes": 20,
}


DATASET_ID_LIST = [
    # "hopper_random",
    # "hopper_replay",
    # "hopper_medium_replay",
    # "hopper_medium_d4rl",
    # "hopper_expert_d4rl",
    # "hopper_medium_expert",
    "hopper_simple",
    # "hopper_medium",
    # "hopper_expert",
    # "walker2d_random",
    # "walker2d_replay",
    # "walker2d_medium_replay",
    # "walker2d_medium_d4rl",
    # "walker2d_expert_d4rl",
    # "walker2d_medium_expert",
    # "walker2d_simple",
    # "walker2d_medium",
    # "walker2d_expert",
    # "halfcheetah_random",
    # "halfcheetah_replay",
    # "halfcheetah_medium_replay",
    # "halfcheetah_medium_d4rl",
    # "halfcheetah_expert_d4rl",
    # "halfcheetah_medium_expert",
    # "halfcheetah_simple",
    # "halfcheetah_medium",
    # "halfcheetah_expert",
]


AGENT_ID_LIST = [
    # "bc_deterministic",
    # "bc_stochastic",
    # "td3bc",
    # "iql",
    # "cql",
    # "cql_max_q",
    # "cql_soft_q",
    "aspl",
    # "sdc_cql",
    # "sdc_pre",
    # "scas_min",
    # "scas_mean",
    # "scas_aspl",
]


TASK_LIST = [
    ["hopper_simple", "aspl", {"alpha": 0.5}, {"steps": 200_000}],
    ["hopper_medium", "aspl", {"alpha": 0.5}, {"steps": 200_000}],
    ["hopper_expert", "aspl", {"alpha": 0.5}, {"steps": 200_000}],
    # ["hopper_simple", "cql_soft_q", {"threshold": 1.5}, {"steps": 1_000_000}],
    # ["hopper_medium", "cql_soft_q", {"threshold": 1.0}, {"steps": 500_000}],
    # ["hopper_expert", "cql_soft_q", {"threshold": 0.5}, {"steps": 500_000}],
]


def normalize_tasks() -> list[list]:
    if TASK_LIST:
        return TASK_LIST
    return [
        [dataset_id, agent_id, {}, {}]
        for dataset_id in DATASET_ID_LIST
        for agent_id in AGENT_ID_LIST
    ]


def view_train(path, index: int, dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    returns_output_path = returns_path("train", task_id)
    steps_output_path = steps_path("train", task_id)
    metrics_path = metric_path(task_id)
    output_path = plot_path(index, dataset_id, agent_id)

    print(f"plot dataset={dataset_id}, agent={agent_id}")
    cal_returns(path, returns_output_path)
    cal_steps(path, steps_output_path)
    plot([metrics_path], [returns_output_path, steps_output_path], output_path)
    print(f"saved: {output_path}")


def main() -> None:
    for index, (dataset_id, agent_id, agent_config, task_train_kwargs) in enumerate(normalize_tasks(), start=1):
        train_kwargs = TRAIN_KWARGS | task_train_kwargs
        train_kwargs = {key: value for key, value in train_kwargs.items() if value is not None}
        start = train_kwargs.get("start", 0)
        dataset = make_dataset(dataset_id, device="cuda")
        agent = make_agent(agent_id, dataset, device="cuda", **agent_config)

        task_id = _task_id(dataset.id, agent.id)
        if start > 0:
            agent.load(task_id, start)
        print(
            f"task={task_id}, dataset={dataset.id}, agent={agent.id}, "
            f"agent_config={agent_config}"
        )
        path = train(
            agent=agent,
            dataset=dataset,
            task_id=task_id,
            **train_kwargs,
        )
        print(f"saved: {path}")
        view_train(path, index, dataset_id, agent_id)


if __name__ == "__main__":
    main()
