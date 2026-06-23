from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import test
from table import plot_table

TASKS = [
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "bc_deterministic", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "bc_deterministic", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "bc_deterministic", {}),
    ({"agent_step": 500_000}, "hopper_replay_medium", {}, "bc_deterministic", {}),
    ({"agent_step": 500_000}, "hopper_replay_expert", {}, "bc_deterministic", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "bc_stochastic", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "bc_stochastic", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "bc_stochastic", {}),
    ({"agent_step": 500_000}, "hopper_replay_medium", {}, "bc_stochastic", {}),
    ({"agent_step": 500_000}, "hopper_replay_expert", {}, "bc_stochastic", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "td3bc", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "td3bc", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "td3bc", {}),
    ({"agent_step": 500_000}, "hopper_replay_medium", {}, "td3bc", {}),
    ({"agent_step": 500_000}, "hopper_replay_expert", {}, "td3bc", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "iql", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "iql", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "iql", {}),
    ({"agent_step": 500_000}, "hopper_replay_medium", {}, "iql", {}),
    ({"agent_step": 500_000}, "hopper_replay_expert", {}, "iql", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "cql", {"threshold": 1.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "cql", {"threshold": 1.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "cql", {"threshold": 1.0}),
    ({"agent_step": 500_000}, "hopper_replay_medium", {}, "cql", {"threshold": 1.5}),
    ({"agent_step": 500_000}, "hopper_replay_expert", {}, "cql", {"threshold": 1.0}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "aspl", {"weight_punish": 0.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "aspl", {"weight_punish": 0.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "aspl", {"weight_punish": 1.0}),
    ({"agent_step": 500_000}, "hopper_replay_medium", {}, "aspl", {"weight_punish": 0.5}),
    ({"agent_step": 500_000}, "hopper_replay_expert", {}, "aspl", {"weight_punish": 1.0}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_medium", {}, "sdc", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_hybrid", {}, "sdc", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_expert", {}, "sdc", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_medium", {}, "sdc", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_expert", {}, "sdc", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_medium", {}, "sdc_cql", {"threshold": 10}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_hybrid", {}, "sdc_cql", {"threshold": 5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_expert", {}, "sdc_cql", {"threshold": 0.5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_medium", {}, "sdc_cql", {"threshold": 5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_expert", {}, "sdc_cql", {"threshold": 0.5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_medium", {}, "scas", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_hybrid", {}, "scas", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_expert", {}, "scas", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_medium", {}, "scas", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_expert", {}, "scas", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_medium", {}, "scas_aspl", {"weight_punish": 0.5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_hybrid", {}, "scas_aspl", {"weight_punish": 0.5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_expert", {}, "scas_aspl", {"weight_punish": 1.0}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_medium", {}, "scas_aspl", {"weight_punish": 0.5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_replay_expert", {}, "scas_aspl", {"weight_punish": 1.0}),
]


def test_agent(
    task_kwargs: dict,
    dataset_id: str,
    env_kwargs: dict,
    agent_id: str,
    agent_kwargs: dict,
) -> None:
    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(
        agent_id,
        dataset,
        device="cuda",
        model_step=task_kwargs.get("model_step"),
        **agent_kwargs,
    )

    task_id = _task_id(dataset.id, agent.id)
    agent_step = task_kwargs.get("agent_step", 0)
    if agent_step > 0:
        agent.load(task_id, agent_step)
    print(
        f"task={task_id}, dataset={dataset.id}, agent={agent.id}, "
        f"agent_kwargs={agent_kwargs}"
    )
    env = dataset.make_env(**env_kwargs)
    path = test(
        task_id,
        agent,
        env,
        episodes=task_kwargs.get("episodes", 100),
        seed=task_kwargs.get("seed", 42),
        print_interval=task_kwargs.get("print_interval", 1),
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    for task_kwargs, dataset_id, env_kwargs, agent_id, agent_kwargs in TASKS:
        test_agent(task_kwargs, dataset_id, env_kwargs, agent_id, agent_kwargs)
    plot_table(list(dict.fromkeys((dataset_id, agent_id) for _, dataset_id, _, agent_id, _ in TASKS)))
