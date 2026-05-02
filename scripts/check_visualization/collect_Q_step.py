import gymnasium as gym
import minari
import minigrid  # noqa: F401
import numpy as np

from ice_offline.agent import StepQAgent
from ice_offline.dataset import OldValueCollector
from ice_offline.dataset.value_oneroom import make_value_env
from ice_offline.tools import print_stage


ENV_ID = "BabyAI-OneRoomS8-v0"
DATASET_ID = "check/visualization-q-step-v0"
MAX_TRAIN_STEPS = 1_000
LOG_EVERY_STEPS = 50
POLICY_EPSILON = 1.0
SEED_BASE = 42


def _obs_key(observation: dict[str, np.ndarray]) -> bytes:
    return np.asarray(observation["image"], dtype=np.uint8).tobytes()


def main(
    env_id: str = ENV_ID,
    dataset_id: str = DATASET_ID,
    max_train_steps: int = MAX_TRAIN_STEPS,
    policy_epsilon: float = POLICY_EPSILON,
    log_every_steps: int = LOG_EVERY_STEPS,
) -> None:
    print_stage("Init")
    agent = StepQAgent(n_actions=4, encoder=_obs_key, seed=SEED_BASE)
    eval_env = gym.make(env_id)
    base_env = make_value_env()
    value_collector = OldValueCollector(
        base_env,
        lambda observation, action: agent.Q(observation, int(action)),
    )
    collector = minari.DataCollector(value_collector, record_infos=False)

    total_steps = 0
    episode = 0

    try:
        print_stage("Train + Record")
        while total_steps < max_train_steps:
            episode += 1
            observation, _ = collector.reset(seed=SEED_BASE + episode)
            value_collector.record()
            done = False
            truncated = False

            while not (done or truncated) and total_steps < max_train_steps:
                action = agent.policy(observation, epsilon=policy_epsilon)
                next_observation, _, done, truncated, _ = collector.step(action)
                agent.update(observation, action)
                value_collector.record()
                observation = next_observation
                total_steps = agent.step_count()

                if total_steps % log_every_steps == 0:
                    print(f"step={total_steps}/{max_train_steps} episode={episode} q_states={agent.q_states()}")

        print_stage("Save Dataset")
        try:
            minari.delete_dataset(dataset_id)
        except Exception:
            pass

        collector.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="q_step",
            author="local_test",
            author_email="local_test@example.com",
            code_permalink="https://example.com/check_visualization/collect_Q_step",
            eval_env=eval_env,
            description="Collect one-room value_data with step-index Q agent for visualization checks",
        )
        value_path = value_collector.save(dataset_id)

        print_stage("Verify Dataset")
        dataset = minari.load_dataset(dataset_id)
        print(f"dataset_id={dataset.spec.dataset_id}")
        print(f"episodes={dataset.total_episodes} total_steps={dataset.total_steps}")
        print(f"train_steps={total_steps} q_states={agent.q_states()}")
        print(f"value_data_path={value_path}")
    finally:
        collector.close()
        eval_env.close()


if __name__ == "__main__":
    main()
