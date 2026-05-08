from typing import Any

import minari
import numpy as np

from ice_offline.agent import QTableAgent
from ice_offline.pipeline import ValueLoader
from ice_offline.pipeline.value_oneroom import ACTIONS, ValueOneRoomCollector, make_value_env
from ice_offline.tools import print_stage


# ====================
# config
# ====================
DATASET_ID = "check/compare_value-v0"
MAX_EPISODES = 5
SEED_BASE = 42
POLICY_EPSILON = 0.3

def _obs_key(observation: Any) -> tuple[bytes, int]:
    return (np.asarray(observation["image"], dtype=np.uint8).tobytes(), int(observation["direction"]))

def eval_fn(agent_ref: QTableAgent, o: Any, a: int) -> float:
    return float(agent_ref.Q(o, a))

def main() -> None:
    print_stage("Collect")
    agent = QTableAgent(
        n_actions=len(ACTIONS),
        encoder=_obs_key,
        alpha=0.1,
        gamma=0.99,
        seed=SEED_BASE,
    )

    eval_env = make_value_env()
    base_env = make_value_env()
    value_collector = ValueOneRoomCollector(
        base_env,
        eval_fn=eval_fn,
        agent=agent,
    )
    collector = minari.DataCollector(value_collector, record_infos=False)

    expected_episodes: list[list[np.ndarray]] = []
    total_steps = 0

    try:
        for episode in range(1, MAX_EPISODES + 1):
            observation, _ = collector.reset(seed=SEED_BASE + episode)
            expected_steps: list[np.ndarray] = [value_collector.eval().copy()]
            done = False
            truncated = False
            episode_steps = 0

            while not (done or truncated):
                action = int(agent.policy(observation, epsilon=POLICY_EPSILON))
                next_observation, reward, done, truncated, _ = collector.step(action)
                agent.update(observation, action, float(reward), next_observation, bool(done or truncated))
                expected_steps.append(value_collector.eval().copy())
                observation = next_observation
                episode_steps += 1
                total_steps += 1

            expected_episodes.append(expected_steps)
            print(f"episode={episode} episode_steps={episode_steps} done={done} truncated={truncated}")

        try:
            minari.delete_dataset(DATASET_ID)
        except Exception:
            pass

        collector.create_dataset(
            dataset_id=DATASET_ID,
            algorithm_name="random_policy_value",
            author="local_test",
            author_email="local_test@example.com",
            code_permalink="https://example.com/compare_value",
            eval_env=eval_env,
            description="collect and compare value data with loader",
        )
        value_data_path = value_collector.save(DATASET_ID)
        print(f"value_data_path={value_data_path}")
    finally:
        eval_env.close()
        collector.close()

    print(f"total_steps={total_steps}")

    print_stage("Compare")
    value_loader = ValueLoader(DATASET_ID)
    try:
        loaded_episode_count = value_loader.get_episode_count()
        expected_episode_count = len(expected_episodes)
        if loaded_episode_count != expected_episode_count:
            raise SystemExit(
                f"FAIL: episode_count mismatch expected={expected_episode_count} actual={loaded_episode_count}"
            )

        for episode_index, expected_steps in enumerate(expected_episodes):
            actual_steps = value_loader.load_episode(episode_index)
            if len(actual_steps) != len(expected_steps):
                raise SystemExit(
                    f"FAIL: step_count mismatch episode={episode_index} "
                    f"expected={len(expected_steps)} actual={len(actual_steps)}"
                )
            for step_index, (expected_step, actual_step) in enumerate(zip(expected_steps, actual_steps)):
                if expected_step.shape != actual_step.shape:
                    raise SystemExit(
                        f"FAIL: shape mismatch episode={episode_index} step={step_index} "
                        f"expected={tuple(expected_step.shape)} actual={tuple(actual_step.shape)}"
                    )
                if not np.array_equal(expected_step, actual_step):
                    raise SystemExit(f"FAIL: value mismatch episode={episode_index} step={step_index}")
    finally:
        value_loader.close()

    print("PASS: compare_value all matched")


if __name__ == "__main__":
    main()
