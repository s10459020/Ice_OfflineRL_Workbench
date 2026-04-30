
import gymnasium as gym
import minari
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.env.common import MissionTextWrapper, NoJpegImageWrapper, insert_render_quiet_innermost
from ice_offline.env.common.step_penalty_wrapper import StepPenaltyWrapper
from ice_offline.env.replay import StateCollector, ValueCollector
from ice_offline.tools import print_stage

ENV_ID = "BabyAI-OneRoomS8-v0"
DATASET_ID = "test_agent_collect-v0"
MODEL_ID = f"{ENV_ID}_QTableAgent"
MODEL_STEP = 1000000

MAX_STEPS = 1_000
LOG_EVERY_STEPS = 50


def make_collect_env(env_id: str, agent: QTableAgent):
    env = gym.make(env_id)
    env = insert_render_quiet_innermost(env)
    env = FullyObsWrapper(env)
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    env = StepPenaltyWrapper(env, step_penalty=0.01)
    state_collector = StateCollector(env)
    value_collector = ValueCollector(state_collector, lambda obs, action: agent.Q(obs, action))
    collector = minari.DataCollector(value_collector, record_infos=False)
    eval_env = gym.make(env_id)
    return collector, state_collector, value_collector, eval_env


def main() -> None:
    print_stage("Init")
    agent = QTableAgent.load(
        model_id=MODEL_ID,
        step=MODEL_STEP,
        encoder=lambda obs: np.asarray(obs["image"], dtype=np.uint8).tobytes(),
    )
    collector, state_collector, value_collector, eval_env = make_collect_env(ENV_ID, agent)

    total_steps = 0
    episode = 0
    try:
        print_stage("Collect + Record")
        while total_steps < MAX_STEPS:
            episode += 1
            obs, _ = collector.reset(seed=42 + episode)
            value_collector.record()
            done = False
            truncated = False

            while not (done or truncated) and total_steps < MAX_STEPS:
                action = int(agent.policy(obs))
                next_obs, _, done, truncated, _ = collector.step(action)
                value_collector.record()
                obs = next_obs
                total_steps += 1

                if total_steps % LOG_EVERY_STEPS == 0:
                    print(f"step={total_steps}/{MAX_STEPS} episode={episode}")

        print_stage("Save Dataset")
        try:
            minari.delete_dataset(DATASET_ID)
        except Exception:
            pass

        collector.create_dataset(
            dataset_id=DATASET_ID,
            algorithm_name="q_learning_eval",
            author="local_test",
            author_email="local_test@example.com",
            code_permalink="https://example.com/test_agent_collect",
            eval_env=eval_env,
            description="Load pretrained Q-table and collect state/value dataset",
        )
        state_path = state_collector.save(DATASET_ID)
        value_path = value_collector.save(DATASET_ID)

        print_stage("Verify Dataset")
        dataset = minari.load_dataset(DATASET_ID)
        print(f"dataset_id={dataset.spec.dataset_id}")
        print(f"episodes={dataset.total_episodes} total_steps={dataset.total_steps}")
        print(f"collect_steps={total_steps}")
        print(f"state_data_path={state_path}")
        print(f"value_data_path={value_path}")
    finally:
        collector.close()
        eval_env.close()


if __name__ == "__main__":
    main()
