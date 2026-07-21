from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset
from plot import analyze
from plot import plot
from ice_offline.run.test import test_eval
from view import save_boxplots
from view import save_tables

EXPERIMENT = "noise_init"
EXPERIMENT_TRAIN = "base_train"

DATASETS = [
    ("noise_init_5e-4@walker2d_d4rl_medium", "walker2d_d4rl_medium", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@walker2d_d4rl_medium", "walker2d_d4rl_medium", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@walker2d_d4rl_medium", "walker2d_d4rl_medium", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@walker2d_d4rl_medium", "walker2d_d4rl_medium", {"reset_noise_scale": 5e-1}),
    ("noise_init_5e-4@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", {"reset_noise_scale": 5e-1}),
    ("noise_init_5e-4@walker2d_replay_medium", "walker2d_replay_medium", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@walker2d_replay_medium", "walker2d_replay_medium", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@walker2d_replay_medium", "walker2d_replay_medium", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@walker2d_replay_medium", "walker2d_replay_medium", {"reset_noise_scale": 5e-1}),
]

AGENTS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("aspl_c", None, 500_000),
    ("scas_gp", 100_000, 500_000),
    ("scaspl_n", 100_000, 500_000),
    ("scc_n", 100_000, 500_000),
]

TASKS = [
    (test_dataset_id, train_dataset_id, env_kwargs, agent_id, model_step, agent_step)
    for test_dataset_id, train_dataset_id, env_kwargs in DATASETS
    for agent_id, model_step, agent_step in [
        ("bc", None, 50_000),
        ("iql", None, 200_000),
        ("cql", None, 500_000),
        ("aspl_c", None, 500_000),
        ("scc_n", 100_000, 500_000),
    ]
]

COUNT = 20
EVALS = 100
INTERVAL = 1_000


def _steps(start_step: int) -> list[int]:
    return [start_step + INTERVAL * index for index in range(COUNT + 1)]


def test(
    test_dataset_id: str,
    train_dataset_id: str,
    env_kwargs: dict,
    agent_id: str,
    model_step: int | None,
    start_step: int,
) -> str:
    test_id = experiment_task_id(EXPERIMENT, agent_id, test_dataset_id)
    train_id = experiment_task_id(EXPERIMENT_TRAIN, agent_id, train_dataset_id)
    model_train_id = experiment_task_id(EXPERIMENT_TRAIN, "scas_model", train_dataset_id)
    steps = _steps(start_step)

    dataset = make_dataset(train_dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step, model_train_id=model_train_id)
    print(f"task={test_id}, train_id={train_id}, reset_noise_scale={env_kwargs['reset_noise_scale']:g}")
    path = test_eval(
        test_id,
        train_id,
        agent,
        dataset.make_env(**env_kwargs),
        steps,
        episodes=EVALS,
    )
    print(f"saved: {path}")
    return test_id


if __name__ == "__main__":
    test_tasks = TASKS or [
        (test_dataset_id, train_dataset_id, env_kwargs, agent_id, model_step, agent_step)
        for test_dataset_id, train_dataset_id, env_kwargs in DATASETS
        for agent_id, model_step, agent_step in AGENTS
    ]

    for test_dataset_id, train_dataset_id, env_kwargs, agent_id, model_step, agent_step in test_tasks:
        test_id = test(
            test_dataset_id,
            train_dataset_id,
            env_kwargs,
            agent_id,
            model_step,
            agent_step,
        )
        analyze(test_id, eval_path(test_id))
        plot(test_id, returns_path(test_id), test_dataset_id, agent_id)

    dataset_ids = [dataset_id for dataset_id, _, _ in DATASETS]
    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(dataset_ids, agent_ids)
    save_boxplots(dataset_ids, agent_ids)
