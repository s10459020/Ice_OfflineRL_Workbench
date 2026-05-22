
import gymnasium as gym
import minari
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent.q_table import QTableAgent
from ice_offline.env.common.mission_text_wrapper import MissionTextWrapper`nfrom ice_offline.env.common.no_jpeg_image_wrapper import NoJpegImageWrapper
from ice_offline.pipeline.state_collector import StateCollector`nfrom ice_offline.pipeline.value_collector import ValueCollector
from ice_offline.tools.printer import print_stage


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    return env


def main(
    env_id: str,
    dataset_id: str,
    max_train_steps: int = 1_000,
    policy_epsilon: float = 1,
    log_every_steps: int = 50,
) -> None:
    print_stage("Init")
    eval_env = gym.make(env_id)
    agent = QTableAgent(
        n_actions=4,
        encoder=lambda obs: np.asarray(obs["image"], dtype=np.uint8).tobytes(),
        alpha=1.0,
        gamma=0.99,
        seed=42,
    )

    env = make_env(env_id)
    state_collector = StateCollector(env)
    value_collector = ValueCollector(
        state_collector,
        lambda obs, action: float(agent.Q(obs, int(action))),
    )
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
                next_obs, _, done, truncated, _ = collector.step(action)
                agent.update(obs, action, float(total_steps + 1), next_obs, True)
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
                algorithm_name="q_learning_step_value",
                author="local_test",
                author_email="local_test@example.com",
                code_permalink="https://example.com/train_step_collect",
                eval_env=eval_env,
                description="Train Q-table from scratch with Q(s,a)=step update and collect state/value dataset",
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
    dataset_id = "train_Step_collect-v0"

    main(
        env_id=env_id,
        dataset_id=dataset_id,
    )



