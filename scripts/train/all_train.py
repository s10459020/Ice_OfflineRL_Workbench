from ice_offline.agent._lookup import get_agent_train_bundle
from ice_offline.dataset._lookup import DATASET_ONEROOMS8_FULLOBS_OPTIMAL
from ice_offline.dataset._lookup import get_dataset_train_bundle
from ice_offline.runner import TorchBatchOfflineRunner
from ice_offline.tools.printer import print_stage


BATCH_SIZE = 64
TRAIN_STEPS = 100_000
EVAL_INTERVAL = 2_000
EVAL_BATCHES = 8
EVAL_EPISODES = 3
MODEL_LOAD_STEP = 0
MODEL_SAVE_INTERVAL = 50_000


def run_train(dataset_id: str, agent_id: str) -> None:
    dataset, eval_online_fns = get_dataset_train_bundle(dataset_id)
    agent, eval_offline_fns = get_agent_train_bundle(agent_id)

    runner_id = f"{agent_id}__{dataset_id}".replace("/", "__")
    runner = TorchBatchOfflineRunner(
        runner_id=runner_id,
        batch_size=BATCH_SIZE,
        steps=TRAIN_STEPS,
        steps_begin=MODEL_LOAD_STEP,
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
    )


def main() -> None:
    train_pairs: list[tuple[str, str]] = [
        (DATASET_ONEROOMS8_FULLOBS_OPTIMAL, "bc_discrete"),
    ]
    for dataset_id, agent_id in train_pairs:
        run_train(dataset_id=dataset_id, agent_id=agent_id)
    print_stage("Done")


if __name__ == "__main__":
    main()
