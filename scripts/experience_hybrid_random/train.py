from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train
from plot import plot_agent

TASKS = [
    # ([None, 0, 500_000], "hopper_random_expert_3", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_5", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_7", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_9", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_3", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_5", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_7", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_9", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_random_expert_3", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_random_expert_5", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_random_expert_7", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_random_expert_9", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_random_expert_3", "iql", {}),
    # ([None, 0, 500_000], "hopper_random_expert_5", "iql", {}),
    # ([None, 0, 500_000], "hopper_random_expert_7", "iql", {}),
    # ([None, 0, 500_000], "hopper_random_expert_9", "iql", {}),
    ([None, 0, 500_000], "hopper_random_expert_3", "cql", {"threshold": 1.0}),
    ([None, 0, 500_000], "hopper_random_expert_5", "cql", {"threshold": 1.0}),
    ([None, 0, 500_000], "hopper_random_expert_7", "cql", {"threshold": 1.0}),
    ([None, 0, 500_000], "hopper_random_expert_9", "cql", {"threshold": 1.0}),
    ([None, 0, 500_000], "hopper_random_expert_3", "aspl", {"weight_punish": 1.0}),
    ([None, 0, 500_000], "hopper_random_expert_5", "aspl", {"weight_punish": 1.0}),
    ([None, 0, 500_000], "hopper_random_expert_7", "aspl", {"weight_punish": 1.0}),
    ([None, 0, 500_000], "hopper_random_expert_9", "aspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_random_expert_3", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_5", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_7", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_9", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_3", "sdc_cql", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_5", "sdc_cql", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_7", "sdc_cql", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_9", "sdc_cql", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_3", "scas", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_5", "scas", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_7", "scas", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_9", "scas", {}),
    # ([100_000, 0, 500_000], "hopper_random_expert_3", "scas_aspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_random_expert_5", "scas_aspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_random_expert_7", "scas_aspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_random_expert_9", "scas_aspl", {"weight_punish": 1.0}),
]

DATASETS = [
    "hopper_random_expert_3",
    "hopper_random_expert_5",
    "hopper_random_expert_7",
    "hopper_random_expert_9",
]


def train_agent(
    task_steps: list[int | None],
    dataset_id: str,
    agent_id: str,
    agent_kwargs: dict,
) -> None:
    model_start, agent_start, steps = task_steps
    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_start, **agent_kwargs)
    task_id = _task_id(dataset.id, agent.id)

    if agent_start > 0:
        agent.load(task_id, agent_start)

    path = train(
        agent=agent,
        dataset=dataset,
        task_id=task_id,
        eval_env=dataset.make_eval_env(),
        steps=steps,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    for task_steps, dataset_id, agent_id, agent_kwargs in TASKS:
        train_agent(task_steps, dataset_id, agent_id, agent_kwargs)
        plot_agent(DATASETS.index(dataset_id) + 1, dataset_id, agent_id)
