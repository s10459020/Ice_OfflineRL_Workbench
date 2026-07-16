from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import model_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model

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
    ((None, 50_000), "hopper_d4rl_medium", "bc", {}),
    ((None, 50_000), "hopper_d4rl_hybrid", "bc", {}),
    ((None, 50_000), "hopper_d4rl_expert", "bc", {}),
    ((None, 50_000), "hopper_replay_medium", "bc", {}),
    ((None, 50_000), "hopper_replay_expert", "bc", {}),
    ((None, 50_000), "halfcheetah_d4rl_medium", "bc", {}),
    ((None, 50_000), "halfcheetah_d4rl_hybrid", "bc", {}),
    ((None, 50_000), "halfcheetah_d4rl_expert", "bc", {}),
    ((None, 50_000), "halfcheetah_replay_medium", "bc", {}),
    ((None, 50_000), "halfcheetah_replay_expert", "bc", {}),
    ((None, 100_000), "hopper_d4rl_medium", "td3bc_n", {}),
    ((None, 100_000), "hopper_d4rl_hybrid", "td3bc_n", {}),
    ((None, 100_000), "hopper_d4rl_expert", "td3bc_n", {}),
    ((None, 100_000), "hopper_replay_medium", "td3bc_n", {}),
    ((None, 100_000), "hopper_replay_expert", "td3bc_n", {}),
    ((None, 100_000), "halfcheetah_d4rl_medium", "td3bc_n", {}),
    ((None, 100_000), "halfcheetah_d4rl_hybrid", "td3bc_n", {}),
    ((None, 100_000), "halfcheetah_d4rl_expert", "td3bc_n", {}),
    ((None, 100_000), "halfcheetah_replay_medium", "td3bc_n", {}),
    ((None, 100_000), "halfcheetah_replay_expert", "td3bc_n", {}),
    ((None, 200_000), "hopper_d4rl_medium", "iql", {}),
    ((None, 200_000), "hopper_d4rl_hybrid", "iql", {}),
    ((None, 200_000), "hopper_d4rl_expert", "iql", {}),
    ((None, 200_000), "hopper_replay_medium", "iql", {}),
    ((None, 200_000), "hopper_replay_expert", "iql", {}),
    ((None, 200_000), "halfcheetah_d4rl_medium", "iql", {}),
    ((None, 200_000), "halfcheetah_d4rl_hybrid", "iql", {}),
    ((None, 200_000), "halfcheetah_d4rl_expert", "iql", {}),
    ((None, 200_000), "halfcheetah_replay_medium", "iql", {}),
    ((None, 200_000), "halfcheetah_replay_expert", "iql", {}),
    ((None, 500_000), "hopper_d4rl_medium", "cql", {}),
    ((None, 500_000), "hopper_d4rl_hybrid", "cql", {}),
    ((None, 500_000), "hopper_d4rl_expert", "cql", {}),
    ((None, 500_000), "hopper_replay_medium", "cql", {}),
    ((None, 500_000), "hopper_replay_expert", "cql", {}),
    ((None, 500_000), "halfcheetah_d4rl_medium", "cql", {}),
    ((None, 500_000), "halfcheetah_d4rl_hybrid", "cql", {}),
    ((None, 500_000), "halfcheetah_d4rl_expert", "cql", {}),
    ((None, 500_000), "halfcheetah_replay_medium", "cql", {}),
    ((None, 500_000), "halfcheetah_replay_expert", "cql", {}),
    ((None, 500_000), "hopper_d4rl_medium", "aspl_c", {}),
    ((None, 500_000), "hopper_d4rl_hybrid", "aspl_c", {}),
    ((None, 500_000), "hopper_d4rl_expert", "aspl_c", {}),
    ((None, 500_000), "hopper_replay_medium", "aspl_c", {}),
    ((None, 500_000), "hopper_replay_expert", "aspl_c", {}),
    ((None, 500_000), "walker2d_d4rl_medium", "aspl_c", {}),
    ((None, 500_000), "walker2d_d4rl_hybrid", "aspl_c", {}),
    ((None, 500_000), "walker2d_d4rl_expert", "aspl_c", {}),
    ((None, 500_000), "walker2d_replay_medium", "aspl_c", {}),
    ((None, 500_000), "walker2d_replay_expert", "aspl_c", {}),
    ((None, 500_000), "halfcheetah_d4rl_medium", "aspl_c", {}),
    ((None, 500_000), "halfcheetah_d4rl_hybrid", "aspl_c", {}),
    ((None, 500_000), "halfcheetah_d4rl_expert", "aspl_c", {}),
    ((None, 500_000), "halfcheetah_replay_medium", "aspl_c", {}),
    ((None, 500_000), "halfcheetah_replay_expert", "aspl_c", {}),
    ((100_000, 500_000), "hopper_d4rl_medium", "scas_gp", {}),
    ((100_000, 500_000), "hopper_d4rl_hybrid", "scas_gp", {}),
    ((100_000, 500_000), "hopper_d4rl_expert", "scas_gp", {}),
    ((100_000, 500_000), "hopper_replay_medium", "scas_gp", {}),
    ((100_000, 500_000), "hopper_replay_expert", "scas_gp", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_medium", "scas_gp", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_hybrid", "scas_gp", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_expert", "scas_gp", {}),
    ((100_000, 500_000), "halfcheetah_replay_medium", "scas_gp", {}),
    ((100_000, 500_000), "halfcheetah_replay_expert", "scas_gp", {}),
    ((100_000, 500_000), "hopper_d4rl_medium", "scaspl_n", {}),
    ((100_000, 500_000), "hopper_d4rl_hybrid", "scaspl_n", {}),
    ((100_000, 500_000), "hopper_d4rl_expert", "scaspl_n", {}),
    ((100_000, 500_000), "hopper_replay_medium", "scaspl_n", {}),
    ((100_000, 500_000), "hopper_replay_expert", "scaspl_n", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_medium", "scaspl_n", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_hybrid", "scaspl_n", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_expert", "scaspl_n", {}),
    ((100_000, 500_000), "halfcheetah_replay_medium", "scaspl_n", {}),
    ((100_000, 500_000), "halfcheetah_replay_expert", "scaspl_n", {}),
    ((100_000, 500_000), "hopper_d4rl_medium", "scc_gp", {}),
    ((100_000, 500_000), "hopper_d4rl_hybrid", "scc_gp", {}),
    ((100_000, 500_000), "hopper_d4rl_expert", "scc_gp", {}),
    ((100_000, 500_000), "hopper_replay_medium", "scc_gp", {}),
    ((100_000, 500_000), "hopper_replay_expert", "scc_gp", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_medium", "scc_gp", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_hybrid", "scc_gp", {}),
    ((100_000, 500_000), "halfcheetah_d4rl_expert", "scc_gp", {}),
    ((100_000, 500_000), "halfcheetah_replay_medium", "scc_gp", {}),
    ((100_000, 500_000), "halfcheetah_replay_expert", "scc_gp", {}),
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
