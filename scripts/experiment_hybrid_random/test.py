from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import test_eval
from plot import analyze
from plot import plot_test
from train_min import INTERVAL
from view import save_boxplots
from view import save_tables

EXPERIMENT = "experience_hybrid_random"
EXPERIMENT_TRAIN = "experience_hybrid_random_train"

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("aspl_c", None, 500_000),
    ("scas_gp", 500_000, 500_000),
    ("scaspl_n", 500_000, 500_000),
    ("scc_n", 500_000, 500_000),
]

TASKS = []

COUNT = 20
EVALS = 100


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
