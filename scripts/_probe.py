from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.probe import probe
from ice_offline.store.probe.action_axis_probe import ActionAxisProbe

TASKS = [
    # ({"model_step": 10_000, "method": "Pi"}, "hopper_one_simple", {}, "bc_stochastic", {}),
    # ({"model_step": 20_000, "method": "Pi"}, "hopper_simple", {}, "bc_stochastic", {}),
]

TASK_KWARGS = {
    # "model_step": 20_000,
    # "method": "Pi",
    # "seed": None,
}

DATASETS = [
    # ("hopper_simple", {}),
]


AGENTS = [
    # ("bc_stochastic", {}, ["Pi"]),
    # ("td3bc", {}, ["Pi", "Q"]),
    # ("cql_soft_q", {"threshold": 1.5}, ["Pi", "Q"]),
    # ("iql", {}, ["Pi", "Q", "V"]),
]


def normalize_tasks() -> list[tuple[dict[str, object], str, dict[str, object], str, dict[str, object]]]:
    if TASKS:
        return TASKS
    return [
        (
            dict(TASK_KWARGS) | {"method": method},
            dataset_id,
            env_kwargs,
            agent_id,
            agent_kwargs,
        )
        for dataset_id, env_kwargs in DATASETS
        for agent_id, agent_kwargs, methods in AGENTS
        for method in methods
    ]


def main() -> None:
    tasks = normalize_tasks()

    for task_kwargs, dataset_id, env_kwargs, agent_id, agent_kwargs in tasks:
        dataset = make_dataset(dataset_id, device="cuda")
        agent = make_agent(agent_id, dataset, device="cuda", **agent_kwargs)

        model_task_id = _task_id(dataset.id, agent.id)
        method = str(task_kwargs["method"])
        probe_task_id = _task_id(dataset.id, f"{agent.id}_{method.lower()}")
        model_step = task_kwargs.get("model_step", 0)
        if model_step > 0:
            agent.load(model_task_id, model_step)
        print(
            f"task={probe_task_id}, model={model_task_id}, dataset={dataset.id}, "
            f"agent={agent.id}, method={method}, agent_kwargs={agent_kwargs}"
        )
        probe_kwargs: dict[str, object] = {}
        if "episodes" in task_kwargs:
            probe_kwargs["episodes"] = task_kwargs["episodes"]
        if "seed" in task_kwargs:
            probe_kwargs["seed"] = task_kwargs["seed"]
        if env_kwargs:
            probe_kwargs["env_kwargs"] = env_kwargs
        eval_fn = lambda observations, actions: agent.eval(observations, actions, method)
        probe_data = probe(
            probe_task_id,
            dataset,
            ActionAxisProbe(task_kwargs["sample_count"]) if "sample_count" in task_kwargs else ActionAxisProbe(),
            eval_fn,
            **probe_kwargs,
        )
        print(f"saved: {probe_data.path}")


if __name__ == "__main__":
    main()
