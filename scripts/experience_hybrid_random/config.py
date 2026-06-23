DATASETS = [
    "hopper_random_expert_3",
    "hopper_random_expert_5",
    "hopper_random_expert_7",
    "hopper_random_expert_9",
]

MODELS = [
    (100_000, "scas_model", {}),
    (100_000, "sdc_model", {}),
]

AGENTS = [
    ([500_000, 0], "bc_deterministic", {}),
    ([500_000, 0], "bc_stochastic", {}),
    ([500_000, 0], "td3bc", {}),
    ([500_000, 0], "iql", {}),
    ([500_000, 0], "cql", {}),
    ([500_000, 0], "aspl", {}),
    ([500_000, 100_000], "sdc", {}),
    ([500_000, 100_000], "sdc_cql", {}),
    ([500_000, 100_000], "scas", {}),
    ([500_000, 100_000], "scas_aspl", {}),
]

AGENT_TASKS = [
    (task_steps, dataset_id, agent_id, agent_kwargs)
    for dataset_id in DATASETS
    for task_steps, agent_id, agent_kwargs in AGENTS
]

MODEL_TASKS = [
    ({"steps": steps}, dataset_id, model_id, model_kwargs)
    for dataset_id in DATASETS
    for steps, model_id, model_kwargs in MODELS
]
