from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train
from plot import plot_agent

TASKS = [
    # ([None, 0, 500_000], "hopper_d4rl_medium", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_d4rl_hybrid", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_d4rl_expert", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_replay_medium", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_replay_expert", "bc_deterministic", {}),
    # ([None, 0, 500_000], "hopper_d4rl_medium", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_d4rl_hybrid", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_d4rl_expert", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_replay_medium", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_replay_expert", "bc_stochastic", {}),
    # ([None, 0, 500_000], "hopper_d4rl_medium", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_d4rl_hybrid", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_d4rl_expert", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_replay_medium", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_replay_expert", "td3bc", {}),
    # ([None, 0, 500_000], "hopper_d4rl_medium", "iql", {}),
    # ([None, 0, 500_000], "hopper_d4rl_hybrid", "iql", {}),
    # ([None, 0, 500_000], "hopper_d4rl_expert", "iql", {}),
    # ([None, 0, 500_000], "hopper_replay_medium", "iql", {}),
    # ([None, 0, 500_000], "hopper_replay_expert", "iql", {}),
    # ([None, 0, 500_000], "hopper_d4rl_medium", "cql", {"threshold": 1.5}),
    # ([None, 0, 500_000], "hopper_d4rl_hybrid", "cql", {"threshold": 1.5}),
    # ([None, 0, 500_000], "hopper_d4rl_expert", "cql", {"threshold": 1.0}),
    # ([None, 0, 500_000], "hopper_replay_medium", "cql", {"threshold": 1.5}),
    # ([None, 0, 500_000], "hopper_replay_expert", "cql", {"threshold": 1.0}),
    # ([None, 0, 500_000], "hopper_d4rl_medium", "aspl", {"weight_punish": 0.5}),
    # ([None, 0, 500_000], "hopper_d4rl_hybrid", "aspl", {"weight_punish": 0.5}),
    # ([None, 0, 500_000], "hopper_d4rl_expert", "aspl", {"weight_punish": 1.0}),
    # ([None, 0, 500_000], "hopper_replay_medium", "aspl", {"weight_punish": 0.5}),
    # ([None, 0, 500_000], "hopper_replay_expert", "aspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_d4rl_medium", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "sdc", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_medium", "sdc_cql", {"threshold": 10}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "sdc_cql", {"threshold": 5}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "sdc_cql", {"threshold": 5}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "sdc_cql", {"threshold": 5}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "sdc_cql", {"threshold": 5}),
    ([100_000, 0, 500_000], "hopper_d4rl_medium", "scas_lambda_0", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scas_lambda_0", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scas_lambda_0", {}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "scas_lambda_0", {}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scas_lambda_0", {}),
    ([100_000, 0, 500_000], "hopper_d4rl_medium", "scas_lambda_25", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scas_lambda_25", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scas_lambda_25", {}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "scas_lambda_25", {}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scas_lambda_25", {}),
    ([100_000, 0, 500_000], "hopper_d4rl_medium", "scas_lambda_50", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scas_lambda_50", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scas_lambda_50", {}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "scas_lambda_50", {}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scas_lambda_50", {}),
    ([100_000, 0, 500_000], "hopper_d4rl_medium", "scas_lambda_75", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scas_lambda_75", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scas_lambda_75", {}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "scas_lambda_75", {}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scas_lambda_75", {}),
    ([100_000, 0, 500_000], "hopper_d4rl_medium", "scas_lambda_100", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scas_lambda_100", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scas_lambda_100", {}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "scas_lambda_100", {}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scas_lambda_100", {}),
    # ([100_000, 0, 500_000], "hopper_d4rl_medium", "scaspl", {"weight_punish": 0.5}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scaspl", {"weight_punish": 0.5}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scaspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_replay_medium", "scaspl", {"weight_punish": 0.5}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scaspl", {"weight_punish": 1.0}),
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
    dataset_ids = list(dict.fromkeys(dataset_id for _, dataset_id, _, _ in TASKS))
    for task_steps, dataset_id, agent_id, agent_kwargs in TASKS:
        train_agent(task_steps, dataset_id, agent_id, agent_kwargs)
        plot_agent(dataset_ids.index(dataset_id) + 1, dataset_id, agent_id)
