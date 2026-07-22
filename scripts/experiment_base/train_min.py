from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import model_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model

EXPERIMENT_TRAIN = "base_train"

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_expert",
    # "hopper_d4rl_hybrid",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    # "walker2d_d4rl_medium",
    # "walker2d_d4rl_expert",
    # "walker2d_d4rl_hybrid",
    # "walker2d_replay_medium",
    # "walker2d_replay_expert",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_expert",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_replay_medium",
    # "halfcheetah_replay_expert",
]

AGENTS = [
    # ("bc", None, 50_000),
    # ("td3bc_n", None, 100_000),
    # ("iql", None, 200_000),
    # ("cql", None, 500_000),
    # ("aspl_c", None, 500_000),
    # ("scas_gp", 100_000, 500_000),
    # ("scaspl_n", 100_000, 500_000),
    # ("scc_n", 100_000, 500_000),
]

TASKS = [
    ([100_000, 500_000], "hopper_d4rl_medium", "scaspl_n", {}),
    ([100_000, 500_000], "hopper_d4rl_expert", "scaspl_n", {}),
    ([100_000, 500_000], "hopper_d4rl_hybrid", "scaspl_n", {}),
    ([100_000, 500_000], "hopper_replay_medium", "scaspl_n", {}),
    ([100_000, 500_000], "hopper_replay_expert", "scaspl_n", {}),
    ([100_000, 500_000], "halfcheetah_d4rl_medium", "scaspl_n", {}),
    ([100_000, 500_000], "halfcheetah_d4rl_expert", "scaspl_n", {}),
    ([100_000, 500_000], "halfcheetah_d4rl_hybrid", "scaspl_n", {}),
    ([100_000, 500_000], "halfcheetah_replay_medium", "scaspl_n", {}),
    ([100_000, 500_000], "halfcheetah_replay_expert", "scaspl_n", {}),
    ([100_000, 500_000], "hopper_d4rl_medium", "scc_n", {}),
    ([100_000, 500_000], "hopper_d4rl_expert", "scc_n", {}),
    ([100_000, 500_000], "hopper_d4rl_hybrid", "scc_n", {}),
    ([100_000, 500_000], "hopper_replay_medium", "scc_n", {}),
    ([100_000, 500_000], "hopper_replay_expert", "scc_n", {}),
    ([100_000, 500_000], "halfcheetah_d4rl_medium", "scc_n", {}),
    ([100_000, 500_000], "halfcheetah_d4rl_expert", "scc_n", {}),
    ([100_000, 500_000], "halfcheetah_d4rl_hybrid", "scc_n", {}),
    ([100_000, 500_000], "halfcheetah_replay_medium", "scc_n", {}),
    ([100_000, 500_000], "halfcheetah_replay_expert", "scc_n", {}),
]

INTERVAL = 1_000
COUNT = 20


def train_min_agent(
    task_start: list[int | None],
    dataset_id: str,
    agent_id: str,
    agent_kwargs: dict,
) -> None:
    model_start, agent_start = task_start
    model_train_id = experiment_task_id(EXPERIMENT_TRAIN, "scas_model", dataset_id)

    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_start, model_train_id=model_train_id, **agent_kwargs)
    id = experiment_task_id(EXPERIMENT_TRAIN, agent.id, dataset.id)

    if agent_start > 0:
        agent.load(model_path(id, agent_start))

    path = train_model(
        agent=agent,
        dataset=dataset,
        task_id=id,
        start=agent_start,
        steps=agent_start + INTERVAL * COUNT,
        save_interval=INTERVAL,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    tasks = [
        ((model_step, agent_step), dataset_id, agent_id, {})
        for agent_id, model_step, agent_step in AGENTS
        for dataset_id in DATASETS
    ] + TASKS

    for task_start, dataset_id, agent_id, agent_kwargs in tasks:
        train_min_agent(task_start, dataset_id, agent_id, agent_kwargs)
