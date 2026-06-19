from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.eval import cal_returns
from ice_offline.run.eval import cal_steps
from ice_offline.run.plot import plot
from ice_offline.run.train import train

TASKS = [
    # ({"steps": 200_000}, "hopper_simple", {}, "aspl", {"alpha": 0.5}),
    # ({"steps": 200_000}, "hopper_medium", {}, "aspl", {"alpha": 0.5}),
    # ({"steps": 200_000}, "hopper_expert", {}, "aspl", {"alpha": 0.5}),
    # ({"steps": 200_000}, "hopper_simple", {}, "bc_deterministic", {}),
    # ({"steps": 200_000}, "hopper_medium", {}, "bc_deterministic", {}),
    # ({"steps": 200_000}, "hopper_expert", {}, "bc_deterministic", {}),
    # ({"steps": 200_000}, "hopper_simple", {}, "td3bc", {}),
    # ({"steps": 200_000}, "hopper_medium", {}, "td3bc", {}),
    # ({"steps": 200_000}, "hopper_expert", {}, "td3bc", {}),
    # ({"steps": 500_000}, "hopper_simple", {}, "cql_soft_q", {"threshold": 1.5}),
    # ({"steps": 500_000}, "hopper_medium", {}, "cql_soft_q", {"threshold": 1.0}),
    # ({"steps": 500_000}, "hopper_expert", {}, "cql_soft_q", {"threshold": 0.5}),
    # ({"steps": 500_000}, "hopper_simple", {}, "sdc_cql", {"threshold": 3}),
    ({"steps": 500_000}, "hopper_medium", {}, "sdc_cql", {"threshold": 5}),
    # ({"steps": 500_000}, "hopper_expert", {}, "sdc_cql", {"threshold": 0.5}),
]

TASK_KWARGS = {
    # "start": 500_000,
    # "steps": 200_000,
    # "save_interval": 20_000,
    # "eval_interval": 2_000,
    # "print_interval": 200,
    # "eval_episodes": 20,
}

DATASETS = [
    # ("hopper_simple", {}),
    # ("hopper_medium", {}),
    # ("hopper_expert", {}),
]


AGENTS = [
    # ("bc_deterministic", {}),
    # ("bc_stochastic", {}),
    # ("td3bc", {}),
    # ("iql", {}),
    # ("cql", {}),
    # ("cql_max_q", {}),
    # ("cql_soft_q", {"threshold": 1.5}),
    # ("aspl", {"alpha": 0.5}),
    # ("sdc_cql", {"threshold": 2}),
    # ("sdc_pre", {}),
    # ("scas_min", {}),
    # ("scas_mean", {}),
    # ("scas_aspl", {}),
]


def normalize_tasks() -> list[tuple[dict[str, object], str, dict[str, object], str, dict[str, object]]]:
    if TASKS:
        return TASKS
    return [
        (
            dict(TASK_KWARGS),
            dataset_id,
            dataset_kwargs,
            agent_id,
            agent_kwargs,
        )
        for dataset_id, dataset_kwargs in DATASETS
        for agent_id, agent_kwargs in AGENTS
    ]


def view_train(path, index: int, dataset_id: str, agent_id: str) -> None:
    task_id = _task_id(dataset_id, agent_id)
    returns_output_path = returns_path("train", task_id)
    steps_output_path = steps_path("train", task_id)
    metrics_output_path = metric_path(task_id)
    output_path = plot_path(index, dataset_id, agent_id)

    print(f"plot dataset={dataset_id}, agent={agent_id}")
    cal_returns(path, returns_output_path)
    cal_steps(path, steps_output_path)
    plot([metrics_output_path], [returns_output_path, steps_output_path], output_path)
    print(f"saved: {output_path}")


def main() -> None:
    tasks = normalize_tasks()
    dataset_ids: list[str] = []
    for _, dataset_id, _, _, _ in tasks:
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

    for task_kwargs, dataset_id, dataset_kwargs, agent_id, agent_kwargs in tasks:
        dataset = make_dataset(dataset_id, device="cuda")
        agent = make_agent(agent_id, dataset, device="cuda", **agent_kwargs)

        task_id = _task_id(dataset.id, agent.id)
        start = task_kwargs.get("start", 0)
        if start > 0:
            agent.load(task_id, start)
        print(
            f"task={task_id}, dataset={dataset.id}, agent={agent.id}, "
            f"agent_kwargs={agent_kwargs}"
        )
        path = train(
            agent=agent,
            dataset=dataset,
            task_id=task_id,
            **task_kwargs,
        )
        print(f"saved: {path}")
        view_train(path, dataset_ids.index(dataset_id) + 1, dataset_id, agent_id)


if __name__ == "__main__":
    main()
