from ice_offline.agent._lookup import get_agent_train_bundle
from ice_offline.dataset._lookup import get_dataset_train_bundle
from ice_offline.runner import TorchBatchOfflineRunner
from ice_offline.tools.printer import print_stage


BATCH_SIZE = 64
TRAIN_STEPS = 100_000
EVAL_INTERVAL = 500
EVAL_BATCHES = 8
EVAL_EPISODES = 3
MODEL_LOAD_STEP = 0
MODEL_LOAD_AUTO = True
MODEL_SAVE_INTERVAL = 2_000


def run_train(dataset_id: str, agent_id: str) -> None:
    dataset, eval_online_fns, early_stop_events = get_dataset_train_bundle(dataset_id)
    agent, eval_offline_fns = get_agent_train_bundle(agent_id)

    runner_id = f"{dataset_id}__{agent_id}".replace("/", "__")
    runner = TorchBatchOfflineRunner(
        runner_id=runner_id,
        batch_size=BATCH_SIZE,
        steps=TRAIN_STEPS,
        steps_begin=MODEL_LOAD_STEP,
        steps_begin_auto=MODEL_LOAD_AUTO,
        save_interval=MODEL_SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        eval_batches=EVAL_BATCHES,
        eval_episodes=EVAL_EPISODES,
    )

    print_stage("Train")
    runner.train(
        dataset=dataset,
        agent=agent,
        eval_offline_fns=eval_offline_fns,
        eval_online_fns=eval_online_fns,
        early_stop_events=early_stop_events,
    )


def main() -> None:
    train_pairs: list[tuple[str, str]] = [
        ("halfcheetah_simple", "bc_deterministic"),
        ("halfcheetah_simple", "iql_continuous"),
        ("halfcheetah_simple", "cql_continuous"),
        ("halfcheetah_medium", "bc_deterministic"),
        ("halfcheetah_medium", "iql_continuous"),
        ("halfcheetah_medium", "cql_continuous"),
        ("halfcheetah_expert", "bc_deterministic"),
        ("halfcheetah_expert", "iql_continuous"),
        ("halfcheetah_expert", "cql_continuous"),
        ("hopper_simple", "bc_deterministic"),
        ("hopper_simple", "iql_continuous"),
        ("hopper_simple", "cql_continuous"),
        ("hopper_medium", "bc_deterministic"),
        ("hopper_medium", "iql_continuous"),
        ("hopper_medium", "cql_continuous"),
        ("hopper_expert", "bc_deterministic"),
        ("hopper_expert", "iql_continuous"),
        ("hopper_expert", "cql_continuous"),
        ("walker2d_simple", "bc_deterministic"),
        ("walker2d_simple", "iql_continuous"),
        ("walker2d_simple", "cql_continuous"),
        ("walker2d_medium", "bc_deterministic"),
        ("walker2d_medium", "iql_continuous"),
        ("walker2d_medium", "cql_continuous"),
        ("walker2d_expert", "bc_deterministic"),
        ("walker2d_expert", "iql_continuous"),
        ("walker2d_expert", "cql_continuous"),
        ("invertedpendulum_expert", "bc_deterministic"),
        ("invertedpendulum_expert", "iql_continuous"),
        ("invertedpendulum_expert", "cql_continuous"),
    ]
    for dataset_id, agent_id in train_pairs:
        run_train(dataset_id=dataset_id, agent_id=agent_id)
    print_stage("Done")


if __name__ == "__main__":
    main()
