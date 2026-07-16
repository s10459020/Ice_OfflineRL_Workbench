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
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    # "walker2d_d4rl_medium",
    # "walker2d_d4rl_hybrid",
    # "walker2d_d4rl_expert",
    # "walker2d_replay_medium",
    # "walker2d_replay_expert",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_d4rl_expert",
    # "halfcheetah_replay_medium",
    # "halfcheetah_replay_expert",
]

AGENTS = [
    # ("bc", None, 50_000),
    # ("td3_s", None, 100_000),
    # ("td3bc", None, 100_000),
    # ("td3bc_b", None, 100_000),
    # ("td3bc_bgp", None, 100_000),
    # ("td3bc_gp", None, 100_000),
    # ("td3bc_n", None, 100_000),
    # ("td3bc_n_1", None, 100_000),
    # ("iql", None, 200_000),
    # ("cql", None, 500_000),
    # ("aspl_gp", None, 500_000),
    # ("scas_adject", 100_000, 500_000),
    # ("scas_adject_01", 100_000, 500_000),
    # ("scas_adject_00075_00025", 100_000, 500_000),
    # ("scas_adject_075_025", 100_000, 500_000),
    # ("scas_adject_75_25", 100_000, 500_000),
    # ("scas_adject_1", 100_000, 500_000),
    # ("scas_adject_1_01", 100_000, 500_000),
    # ("scas_adject_5_5", 100_000, 500_000),
    # ("scas_adject_10", 100_000, 500_000),
    # ("scas_n", 100_000, 500_000),
    # ("scas_gp", 100_000, 500_000),
    # ("scaspl_n", 100_000, 500_000),
    # ("scaspl_c", 100_000, 500_000),
    # ("scaspl_gp", 100_000, 500_000),
    # ("scaspl_ns", 100_000, 500_000),
    # ("scaspl_nc", 100_000, 500_000),
    # ("scc_n", 100_000, 500_000),
    # ("scc_gp", 100_000, 500_000),
    # ("scc_ns", 100_000, 500_000),
]

TASKS = [
    ("hopper_d4rl_medium", "bc", None, 50_000),
    ("hopper_d4rl_hybrid", "bc", None, 50_000),
    ("hopper_d4rl_expert", "bc", None, 50_000),
    ("hopper_replay_medium", "bc", None, 50_000),
    ("hopper_replay_expert", "bc", None, 50_000),
    ("halfcheetah_d4rl_medium", "bc", None, 50_000),
    ("halfcheetah_d4rl_hybrid", "bc", None, 50_000),
    ("halfcheetah_d4rl_expert", "bc", None, 50_000),
    ("halfcheetah_replay_medium", "bc", None, 50_000),
    ("halfcheetah_replay_expert", "bc", None, 50_000),
    ("hopper_d4rl_medium", "td3bc_n", None, 100_000),
    ("hopper_d4rl_hybrid", "td3bc_n", None, 100_000),
    ("hopper_d4rl_expert", "td3bc_n", None, 100_000),
    ("hopper_replay_medium", "td3bc_n", None, 100_000),
    ("hopper_replay_expert", "td3bc_n", None, 100_000),
    ("halfcheetah_d4rl_medium", "td3bc_n", None, 100_000),
    ("halfcheetah_d4rl_hybrid", "td3bc_n", None, 100_000),
    ("halfcheetah_d4rl_expert", "td3bc_n", None, 100_000),
    ("halfcheetah_replay_medium", "td3bc_n", None, 100_000),
    ("halfcheetah_replay_expert", "td3bc_n", None, 100_000),
    ("hopper_d4rl_medium", "iql", None, 200_000),
    ("hopper_d4rl_hybrid", "iql", None, 200_000),
    ("hopper_d4rl_expert", "iql", None, 200_000),
    ("hopper_replay_medium", "iql", None, 200_000),
    ("hopper_replay_expert", "iql", None, 200_000),
    ("halfcheetah_d4rl_medium", "iql", None, 200_000),
    ("halfcheetah_d4rl_hybrid", "iql", None, 200_000),
    ("halfcheetah_d4rl_expert", "iql", None, 200_000),
    ("halfcheetah_replay_medium", "iql", None, 200_000),
    ("halfcheetah_replay_expert", "iql", None, 200_000),
    ("hopper_d4rl_medium", "cql", None, 500_000),
    ("hopper_d4rl_hybrid", "cql", None, 500_000),
    ("hopper_d4rl_expert", "cql", None, 500_000),
    ("hopper_replay_medium", "cql", None, 500_000),
    ("hopper_replay_expert", "cql", None, 500_000),
    ("halfcheetah_d4rl_medium", "cql", None, 500_000),
    ("halfcheetah_d4rl_hybrid", "cql", None, 500_000),
    ("halfcheetah_d4rl_expert", "cql", None, 500_000),
    ("halfcheetah_replay_medium", "cql", None, 500_000),
    ("halfcheetah_replay_expert", "cql", None, 500_000),
    ("hopper_d4rl_medium", "aspl_c", None, 500_000),
    ("hopper_d4rl_hybrid", "aspl_c", None, 500_000),
    ("hopper_d4rl_expert", "aspl_c", None, 500_000),
    ("hopper_replay_medium", "aspl_c", None, 500_000),
    ("hopper_replay_expert", "aspl_c", None, 500_000),
    ("walker2d_d4rl_medium", "aspl_c", None, 500_000),
    ("walker2d_d4rl_hybrid", "aspl_c", None, 500_000),
    ("walker2d_d4rl_expert", "aspl_c", None, 500_000),
    ("walker2d_replay_medium", "aspl_c", None, 500_000),
    ("walker2d_replay_expert", "aspl_c", None, 500_000),
    ("halfcheetah_d4rl_medium", "aspl_c", None, 500_000),
    ("halfcheetah_d4rl_hybrid", "aspl_c", None, 500_000),
    ("halfcheetah_d4rl_expert", "aspl_c", None, 500_000),
    ("halfcheetah_replay_medium", "aspl_c", None, 500_000),
    ("halfcheetah_replay_expert", "aspl_c", None, 500_000),
    ("hopper_d4rl_medium", "scas_gp", 100_000, 500_000),
    ("hopper_d4rl_hybrid", "scas_gp", 100_000, 500_000),
    ("hopper_d4rl_expert", "scas_gp", 100_000, 500_000),
    ("hopper_replay_medium", "scas_gp", 100_000, 500_000),
    ("hopper_replay_expert", "scas_gp", 100_000, 500_000),
    ("halfcheetah_d4rl_medium", "scas_gp", 100_000, 500_000),
    ("halfcheetah_d4rl_hybrid", "scas_gp", 100_000, 500_000),
    ("halfcheetah_d4rl_expert", "scas_gp", 100_000, 500_000),
    ("halfcheetah_replay_medium", "scas_gp", 100_000, 500_000),
    ("halfcheetah_replay_expert", "scas_gp", 100_000, 500_000),
    ("hopper_d4rl_medium", "scaspl_n", 100_000, 500_000),
    ("hopper_d4rl_hybrid", "scaspl_n", 100_000, 500_000),
    ("hopper_d4rl_expert", "scaspl_n", 100_000, 500_000),
    ("hopper_replay_medium", "scaspl_n", 100_000, 500_000),
    ("hopper_replay_expert", "scaspl_n", 100_000, 500_000),
    ("halfcheetah_d4rl_medium", "scaspl_n", 100_000, 500_000),
    ("halfcheetah_d4rl_hybrid", "scaspl_n", 100_000, 500_000),
    ("halfcheetah_d4rl_expert", "scaspl_n", 100_000, 500_000),
    ("halfcheetah_replay_medium", "scaspl_n", 100_000, 500_000),
    ("halfcheetah_replay_expert", "scaspl_n", 100_000, 500_000),
    ("hopper_d4rl_medium", "scc_gp", 100_000, 500_000),
    ("hopper_d4rl_hybrid", "scc_gp", 100_000, 500_000),
    ("hopper_d4rl_expert", "scc_gp", 100_000, 500_000),
    ("hopper_replay_medium", "scc_gp", 100_000, 500_000),
    ("hopper_replay_expert", "scc_gp", 100_000, 500_000),
    ("halfcheetah_d4rl_medium", "scc_gp", 100_000, 500_000),
    ("halfcheetah_d4rl_hybrid", "scc_gp", 100_000, 500_000),
    ("halfcheetah_d4rl_expert", "scc_gp", 100_000, 500_000),
    ("halfcheetah_replay_medium", "scc_gp", 100_000, 500_000),
    ("halfcheetah_replay_expert", "scc_gp", 100_000, 500_000),
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

    dataset_ids: list[str] = []
    for dataset_id, _, _, _ in test_tasks:
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

    agent_ids: list[str] = []
    for _, agent_id, _, _ in test_tasks:
        if agent_id not in agent_ids:
            agent_ids.append(agent_id)
    save_tables(dataset_ids, agent_ids)
    save_boxplots(dataset_ids, agent_ids)
