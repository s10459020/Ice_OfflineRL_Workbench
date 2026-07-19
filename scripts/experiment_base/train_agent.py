from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import model_path
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_agent
from plot import analyze
from plot import plot_train

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
    # ([None, 0, 50_000], "bc"),
    # ([None, 0, 100_000], "td3bc_n"),
    # ([None, 0, 200_000], "iql"),
    # ([None, 0, 500_000], "cql"),
    # ([None, 0, 500_000], "aspl_c"),
    # ([100_000, 0, 500_000], "scas_gp"),
]

TASKS = [
    ([100_000, 0, 500_000], "halfcheetah_d4rl_medium", "scas_gp", {}),
    ([100_000, 0, 500_000], "halfcheetah_d4rl_hybrid", "scas_gp", {}),
    ([100_000, 0, 500_000], "halfcheetah_d4rl_expert", "scas_gp", {}),
    ([100_000, 0, 500_000], "halfcheetah_replay_medium", "scas_gp", {}),
    ([100_000, 0, 500_000], "halfcheetah_replay_expert", "scas_gp", {}),
]

def train(task_steps: list[int | None], dataset_id: str, agent_id: str, agent_kwargs: dict) -> str:
    id = experiment_task_id(EXPERIMENT_TRAIN, agent_id, dataset_id)
    model_train_id = experiment_task_id(EXPERIMENT_TRAIN, "scas_model", dataset_id)
    model_start, agent_start, steps = task_steps

    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_start, model_train_id=model_train_id, **agent_kwargs)
    if agent_start > 0:
        agent.load(model_path(id, agent_start))

    train_agent(
        agent=agent,
        dataset=dataset,
        task_id=id,
        start=agent_start,
        steps=steps,
    )
    return id

if __name__ == "__main__":
    agent_tasks = [
        (task_steps, dataset_id, agent_id, {})
        for task_steps, agent_id in AGENTS
        for dataset_id in DATASETS
    ] + TASKS

    for task_steps, dataset_id, agent_id, agent_kwargs in agent_tasks:
        id = train(task_steps, dataset_id, agent_id, agent_kwargs)
        analyze(id, eval_path(id))
        plot_train(id, metric_path(id), [returns_path(id), steps_path(id)], dataset_id, agent_id)
