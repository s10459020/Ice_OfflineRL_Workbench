from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import minari
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.env.common import MissionTextWrapper, NoJpegImageWrapper
from ice_offline.env.common.step_penalty_wrapper import StepPenaltyWrapper
from ice_offline.env.replay import StateCollector, ValueCollector
from ice_offline.tools import print_stage


def make_env(env_id: str, step_penalty: float) -> gym.Env:
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    env = StepPenaltyWrapper(env, step_penalty=step_penalty)
    return env


def main(
    env_id: str,
    dataset_id: str,
    max_train_steps: int = 1_000,
    policy_epsilon: float = 0.3,
    log_every_steps: int = 50,
    step_penalty: float = 0.01,
) -> None:
    print_stage("Init")
    eval_env = gym.make(env_id)
    env = make_env(env_id, step_penalty=step_penalty)

    agent = QTableAgent(
        n_actions=4,
        encoder=lambda obs: (int(obs["direction"]), np.asarray(obs["image"], dtype=np.uint8).tobytes()),
        alpha=0.1,
        gamma=0.99,
        seed=42,
    )
    state_collector = StateCollector(env)
    value_collector = ValueCollector(state_collector, lambda obs, action: agent.Q(obs, action))
    collector = minari.DataCollector(value_collector, record_infos=False)

    total_steps = 0
    episode = 0
    state_path: Path | None = None
    value_path: Path | None = None
    try:
        print_stage("Train + Record")
        while total_steps < max_train_steps:
            episode += 1
            obs, _ = collector.reset(seed=42 + episode)
            value_collector.record()
            done = False
            truncated = False
            while not (done or truncated) and total_steps < max_train_steps:
                action = int(agent.policy(obs, epsilon=policy_epsilon))
                next_obs, reward, done, truncated, _ = collector.step(action)
                agent.update(obs, action, float(reward), next_obs, bool(done or truncated))
                value_collector.record()
                obs = next_obs
                total_steps += 1

                if total_steps % log_every_steps == 0:
                    print(f"step={total_steps}/{max_train_steps} episode={episode} q_states={len(agent.Q)}")

        print_stage("Save Dataset")
        try:
            minari.delete_dataset(dataset_id)
        except Exception:
            pass

        try:
            collector.create_dataset(
                dataset_id=dataset_id,
                algorithm_name="q_learning_penalty",
                author="local_test",
                author_email="local_test@example.com",
                code_permalink="https://example.com/train_q_table_penalty_collect",
                eval_env=eval_env,
                description="Train Q-table from scratch with step penalty and collect state/value dataset",
            )
            state_path = state_collector.save(dataset_id)
            value_path = value_collector.save(dataset_id)
        except Exception:
            try:
                minari.delete_dataset(dataset_id)
            except Exception:
                pass
            raise

        print_stage("Verify Dataset")
        dataset = minari.load_dataset(dataset_id)
        print(f"dataset_id={dataset.spec.dataset_id}")
        print(f"episodes={dataset.total_episodes} total_steps={dataset.total_steps}")
        print(f"train_steps={total_steps} q_states={len(agent.Q)}")
        print(f"state_data_path={state_path}")
        print(f"value_data_path={value_path}")
    finally:
        collector.close()
        eval_env.close()


if __name__ == "__main__":
    env_id = "BabyAI-OneRoomS8-v0"
    dataset_id = f"{Path(__file__).stem}-v0"
    max_train_steps = 1_000
    policy_epsilon = 0.3
    log_every_steps = 50
    step_penalty = 0.01

    main(
        env_id=env_id,
        dataset_id=dataset_id,
        max_train_steps=max_train_steps,
        policy_epsilon=policy_epsilon,
        log_every_steps=log_every_steps,
        step_penalty=step_penalty,
    )
