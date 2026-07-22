from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import model_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model

EXPERIMENT_TRAIN = "base_train"
DEVICE = "cpu"

DATASETS = [
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
]

AGENTS = [
    # ("td3", None, 50_000),
    # ("td3_gamma_90", None, 50_000),
    # ("td3_n", None, 50_000),
    # ("td3_r", None, 50_000),
    # ("td3_gp", None, 50_000),
    # ("td3_gpn", None, 50_000),
    # ("td3bc", None, 100_000),
    # ("td3bc_n", None, 100_000),
    # ("td3bc_plus", None, 100_000),
    # ("td3bc_r", None, 100_000),
    # ("td3bc_gp", None, 100_000),
    # ("td3bc_gp_plus", None, 100_000),
    # ("td3bc_gpn", None, 100_000),
    # ("cql", None, 500_000),
    # ("cql_threshold_n5", None, 500_000),
    # ("cql_threshold_5", None, 500_000),
    # ("cql_gp", None, 500_000),
    # ("aspl", None, 200_000),
    # ("aspl_r", None, 200_000),
    # ("aspl_gamma_90", None, 200_000),
    # ("aspl_gamma_95", None, 200_000),
    # ("aspl_gp", None, 1_000_000),
    # ("scas", 100_000, 500_000),
    # ("scas_n", 100_000, 500_000),
    # ("scas_gp", 100_000, 500_000),
    # ("scas_gpn", 100_000, 500_000),
    # ("scaspl", 100_000, 500_000),
    # ("scaspl_n", 100_000, 500_000),
    # ("scaspl_gp", 100_000, 500_000),
    # ("scaspl_c", 100_000, 500_000),
    # ("scaspl_nc", 100_000, 500_000),
    # ("scaspl_gpc", 100_000, 500_000),
    ("scc", 100_000, 500_000),
]

TASKS = [
    # ("walker2d_replay_medium", "scas_gp", 100_000, 500_000),
    # ("walker2d_d4rl_medium", "td3bc_gp", None, 100_000),
    # ("walker2d_d4rl_hybrid", "td3bc_gp", None, 100_000),
    # ("walker2d_d4rl_expert", "td3bc_gp", None, 100_000),
    # ("walker2d_replay_medium", "td3bc_gp", None, 100_000),
    # ("walker2d_replay_expert", "td3bc_gp", None, 100_000),
    # ("walker2d_d4rl_medium", "td3bc_gp_plus", None, 100_000),
    # ("walker2d_d4rl_hybrid", "td3bc_gp_plus", None, 100_000),
    # ("walker2d_d4rl_expert", "td3bc_gp_plus", None, 100_000),
    # ("walker2d_replay_medium", "td3bc_gp_plus", None, 100_000),
    # ("walker2d_replay_expert", "td3bc_gp_plus", None, 100_000),
    # ("walker2d_d4rl_medium", "td3bc_gpn", None, 100_000),
    # ("walker2d_d4rl_hybrid", "td3bc_gpn", None, 100_000),
    # ("walker2d_d4rl_expert", "td3bc_gpn", None, 100_000),
    # ("walker2d_replay_medium", "td3bc_gpn", None, 100_000),
    # ("walker2d_replay_expert", "td3bc_gpn", None, 100_000),
    # ("walker2d_d4rl_medium", "scas_n", 100_000, 500_000),
    # ("walker2d_d4rl_hybrid", "scas_n", 100_000, 500_000),
    # ("walker2d_d4rl_expert", "scas_n", 100_000, 500_000),
    # ("walker2d_replay_medium", "scas_n", 100_000, 500_000),
    # ("walker2d_replay_expert", "scas_n", 100_000, 500_000),
    # ("walker2d_d4rl_medium", "scas_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_hybrid", "scas_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_expert", "scas_gpn", 100_000, 500_000),
    # ("walker2d_replay_medium", "scas_gpn", 100_000, 500_000),
    # ("walker2d_replay_expert", "scas_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_medium", "aspl_gp", None, 500_000),
    # ("walker2d_d4rl_hybrid", "aspl_gp", None, 500_000),
    # ("walker2d_d4rl_expert", "aspl_gp", None, 500_000),
    # ("walker2d_replay_medium", "aspl_gp", None, 500_000),
    # ("walker2d_replay_expert", "aspl_gp", None, 500_000),
]

INTERVAL = 1_000
COUNT = 20


def train_min_agent(
    dataset_id: str,
    agent_id: str,
    model_start: int | None,
    agent_start: int,
) -> None:
    model_train_id = experiment_task_id(EXPERIMENT_TRAIN, "scas_model", dataset_id)

    dataset = make_dataset(dataset_id, device=DEVICE)
    agent = make_agent(agent_id, dataset, device=DEVICE, model_step=model_start, model_train_id=model_train_id)
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
        (dataset_id, agent_id, model_step, agent_step)
        for agent_id, model_step, agent_step in AGENTS
        for dataset_id in DATASETS
    ] + TASKS

    for dataset_id, agent_id, model_step, agent_step in tasks:
        train_min_agent(dataset_id, agent_id, model_step, agent_step)
