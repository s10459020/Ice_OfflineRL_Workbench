from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train
from plot import plot_agent

DATASETS = [
    "hopper_d4rl_medium",
    "hopper_d4rl_hybrid",
    "hopper_d4rl_expert",
    "hopper_replay_medium",
    "hopper_replay_expert",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_d4rl_expert",
    # "halfcheetah_replay_medium",
    # "halfcheetah_replay_expert",
]

AGENTS = [
    # ([None, 0, 50_000], "bc"),
    # ([None, 0, 100_000], "td3"),
    # ([None, 0, 100_000], "td3_gp"),
    # ([None, 0, 100_000], "td3bc"),
    # ([None, 0, 100_000], "td3bc_gp"),
    # ([None, 0, 200_000], "iql"),
    # ([100_000, 0, 100_000], "scas"),
    ([100_000, 0, 500_000], "scas_lambda_0"),
    ([100_000, 0, 500_000], "scas_lambda_25"),
    ([100_000, 0, 500_000], "scas_lambda_50"),
    ([100_000, 0, 500_000], "scas_lambda_75"),
    ([100_000, 0, 500_000], "scas_lambda_100"),
    # ([100_000, 0, 100_000], "scas_gp"),
    # ([100_000, 0, 500_000], "scaspl"),
    # ([100_000, 0, 500_000], "scaspl_gp"),
]

TASKS = [
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
    # ([100_000, 0, 500_000], "hopper_d4rl_medium", "scaspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scaspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scaspl", {"weight_punish": 1.0}),
    ([100_000, 0, 500_000], "hopper_replay_medium", "scaspl", {"weight_correction": 1}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scaspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_d4rl_medium", "scaspl_gp", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_d4rl_hybrid", "scaspl_gp", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "hopper_d4rl_expert", "scaspl_gp", {"weight_punish": 1.0}),
    ([100_000, 0, 500_000], "hopper_replay_medium", "scaspl_gp", {"weight_correction": 1}),
    # ([100_000, 0, 500_000], "hopper_replay_expert", "scaspl_gp", {"weight_punish": 1.0}),
    # ([None, 0, 500_000], "halfcheetah_d4rl_medium", "cql", {"threshold": 1.5}),
    # ([None, 0, 500_000], "halfcheetah_d4rl_hybrid", "cql", {"threshold": 1.5}),
    # ([None, 0, 500_000], "halfcheetah_d4rl_expert", "cql", {"threshold": 1.0}),
    # ([None, 0, 500_000], "halfcheetah_replay_medium", "cql", {"threshold": 1.5}),
    # ([None, 0, 500_000], "halfcheetah_replay_expert", "cql", {"threshold": 1.0}),
    # ([None, 0, 500_000], "halfcheetah_d4rl_medium", "aspl", {"weight_punish": 0.5}),
    # ([None, 0, 500_000], "halfcheetah_d4rl_hybrid", "aspl", {"weight_punish": 0.5}),
    # ([None, 0, 500_000], "halfcheetah_d4rl_expert", "aspl", {"weight_punish": 1.0}),
    # ([None, 0, 500_000], "halfcheetah_replay_medium", "aspl", {"weight_punish": 0.5}),
    # ([None, 0, 500_000], "halfcheetah_replay_expert", "aspl", {"weight_punish": 1.0}),
    # ([100_000, 0, 500_000], "halfcheetah_d4rl_medium", "scaspl", {"weight_punish": 0.5, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_d4rl_hybrid", "scaspl", {"weight_punish": 0.5, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_d4rl_expert", "scaspl", {"weight_punish": 1.0, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_replay_medium", "scaspl", {"weight_correction": 0.5}),
    # ([100_000, 0, 500_000], "halfcheetah_replay_expert", "scaspl", {"weight_punish": 1.0, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_d4rl_medium", "scaspl_gp", {"weight_punish": 0.5, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_d4rl_hybrid", "scaspl_gp", {"weight_punish": 0.5, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_d4rl_expert", "scaspl_gp", {"weight_punish": 1.0, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_replay_medium", "scaspl_gp", {"weight_punish": 0.5, "weight_correction": 0.25}),
    # ([100_000, 0, 500_000], "halfcheetah_replay_expert", "scaspl_gp", {"weight_punish": 1.0, "weight_correction": 0.25}),
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
    tasks = [
        (task_steps, dataset_id, agent_id, {})
        for task_steps, agent_id in AGENTS
        for dataset_id in DATASETS
    ] + TASKS

    for task_steps, dataset_id, agent_id, agent_kwargs in tasks:
        train_agent(task_steps, dataset_id, agent_id, agent_kwargs)
        plot_agent(dataset_id, agent_id)
