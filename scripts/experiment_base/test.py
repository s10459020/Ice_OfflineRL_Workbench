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
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_d4rl_expert",
    # "halfcheetah_replay_medium",
    # "halfcheetah_replay_expert",
]

AGENTS = [
    # ("bc", None, 50_000),
    ("td3bc", None, 100_000),
    ("td3bc_n", None, 100_000),
    ("td3bc_n_1", None, 100_000),
    # ("iql", None, 200_000),
    # ("cql", None, 500_000),
    # ("aspl_gp", None, 500_000),
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
    steps = _steps(start_step)

    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step)
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
    for agent_id, model_step, agent_step in AGENTS:
        for dataset_id in DATASETS:
            id = test(dataset_id, agent_id, model_step, agent_step)
            analyze(id, eval_path(id))
            plot_test(id, returns_path(id), dataset_id, agent_id)

    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(DATASETS, agent_ids)
    save_boxplots(DATASETS, agent_ids)
