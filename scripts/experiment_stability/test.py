from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset
from plot import analyze
from plot import plot_test
from ice_offline.run.test import test_eval
from view import save_boxplots
from view import save_tables

EXPERIMENT = "base"
EXPERIMENT_TRAIN = "base_train"

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
    ("td3bc_plus", None, 100_000),
    # ("td3bc_r", None, 100_000),
    ("td3bc_gp", None, 100_000),
    ("td3bc_gp_plus", None, 100_000),
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
    # ("scaspl_gpn", 100_000, 500_000),
]

TASKS = [
    # ("walker2d_d4rl_medium", "td3bc_gpn", None, 100_000),
    # ("walker2d_d4rl_hybrid", "td3bc_gpn", None, 100_000),
    # ("walker2d_d4rl_expert", "td3bc_gpn", None, 100_000),
    # ("walker2d_replay_medium", "td3bc_gpn", None, 100_000),
    # ("walker2d_replay_expert", "td3bc_gpn", None, 100_000),
    # ("walker2d_d4rl_medium", "aspl", None, 200_000),
    # ("walker2d_d4rl_hybrid", "aspl", None, 200_000),
    # ("walker2d_d4rl_expert", "aspl", None, 200_000),
    # ("walker2d_replay_medium", "aspl", None, 200_000),
    # ("walker2d_replay_expert", "aspl", None, 200_000),
    # ("walker2d_d4rl_medium", "aspl_c", None, 500_000),
    # ("walker2d_d4rl_hybrid", "aspl_c", None, 500_000),
    # ("walker2d_d4rl_expert", "aspl_c", None, 500_000),
    # ("walker2d_replay_medium", "aspl_c", None, 500_000),
    # ("walker2d_replay_expert", "aspl_c", None, 500_000),
    # ("walker2d_d4rl_medium", "scas", 100_000, 500_000),
    # ("walker2d_d4rl_hybrid", "scas", 100_000, 500_000),
    # ("walker2d_d4rl_expert", "scas", 100_000, 500_000),
    # ("walker2d_replay_medium", "scas", 100_000, 500_000),
    # ("walker2d_replay_expert", "scas", 100_000, 500_000),
    # ("walker2d_d4rl_medium", "scas_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_hybrid", "scas_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_expert", "scas_gpn", 100_000, 500_000),
    # ("walker2d_replay_medium", "scas_gpn", 100_000, 500_000),
    # ("walker2d_replay_expert", "scas_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_medium", "scaspl", 100_000, 500_000),
    # ("walker2d_d4rl_hybrid", "scaspl", 100_000, 500_000),
    # ("walker2d_d4rl_expert", "scaspl", 100_000, 500_000),
    # ("walker2d_replay_medium", "scaspl", 100_000, 500_000),
    # ("walker2d_replay_expert", "scaspl", 100_000, 500_000),
    # ("walker2d_d4rl_medium", "scaspl_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_hybrid", "scaspl_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_expert", "scaspl_gpn", 100_000, 500_000),
    # ("walker2d_replay_medium", "scaspl_gpn", 100_000, 500_000),
    # ("walker2d_replay_expert", "scaspl_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_medium", "scc_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_hybrid", "scc_gpn", 100_000, 500_000),
    # ("walker2d_d4rl_expert", "scc_gpn", 100_000, 500_000),
    # ("walker2d_replay_medium", "scc_gpn", 100_000, 500_000),
    # ("walker2d_replay_expert", "scc_gpn", 100_000, 500_000),
]

COUNT = 20
EVALS = 100
INTERVAL = 1_000


def _steps(start_step: int) -> list[int]:
    return [start_step + INTERVAL * index for index in range(COUNT + 1)]


def test(
    dataset_id: str,
    agent_id: str,
    model_step: int | None,
    start_step: int,
) -> str:
    test_id = experiment_task_id(EXPERIMENT, agent_id, dataset_id)
    train_id = experiment_task_id(EXPERIMENT_TRAIN, agent_id, dataset_id)
    model_train_id = experiment_task_id(EXPERIMENT_TRAIN, "scas_model", dataset_id)
    steps = _steps(start_step)

    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step, model_train_id=model_train_id)
    path = test_eval(
        test_id,
        train_id,
        agent,
        dataset.make_env(),
        steps,
        episodes=EVALS,
    )
    print(f"saved: {path}")
    return test_id


if __name__ == "__main__":
    test_tasks = [
        (dataset_id, agent_id, model_step, agent_step)
        for agent_id, model_step, agent_step in AGENTS
        for dataset_id in DATASETS
    ] + TASKS

    for dataset_id, agent_id, model_step, agent_step in test_tasks:
        id = test(dataset_id, agent_id, model_step, agent_step)
        analyze(id, eval_path(id))
        plot_test(id, returns_path(id), dataset_id, agent_id)

    agent_ids: list[str] = []
    for _, agent_id, _, _ in test_tasks:
        if agent_id not in agent_ids:
            agent_ids.append(agent_id)
    save_tables(DATASETS, agent_ids)
    save_boxplots(DATASETS, agent_ids)
