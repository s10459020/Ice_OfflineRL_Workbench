from __future__ import annotations

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


DATASET_ID = "train_agent_q_table_state_value-v0"
MAX_TRAIN_STEPS = 20000
LOG_EVERY_STEPS = 1
POLICY_EPSILON = 0.3


# ====================
# Init
# ====================
print_stage("Init")
eval_env = gym.make("BabyAI-OneRoomS8-v0")
env = gym.make("BabyAI-OneRoomS8-v0")
env = FullyObsWrapper(env)
env = MissionTextWrapper(env)
env = NoJpegImageWrapper(env)
#env = StepPenaltyWrapper(env, step_penalty=0.1)

agent = QTableAgent(
    n_actions=4,
    encoder=lambda obs: (int(obs["direction"]), np.asarray(obs["image"], dtype=np.uint8).tobytes()),
    alpha=0.9,
    gamma=0.99,
    epsilon=0.3,
    seed=42,
)
state_collector = StateCollector(env)
value_collector = ValueCollector(state_collector, lambda obs, action: agent.Q(obs, action))
collector = minari.DataCollector(value_collector, record_infos=False)


# ====================
# Train + Record
# ====================
print_stage("Train + Record")
total_steps = 0
try:
    episode = 0
    while total_steps < MAX_TRAIN_STEPS:
        episode += 1
        obs, _ = collector.reset(seed=42 + episode)
        value_collector.record()
        episode_steps = 0
        done = False
        truncated = False
        while not (done or truncated) and total_steps < MAX_TRAIN_STEPS:
            action = int(agent.policy(obs, epsilon=POLICY_EPSILON))
            next_obs, reward, done, truncated, _ = collector.step(action)
            agent.update(obs, action, float(reward), next_obs, bool(done or truncated))
            value_collector.record()

            obs = next_obs
            episode_steps += 1
            total_steps += 1
            if total_steps % LOG_EVERY_STEPS == 0:
                print(
                    f"step={total_steps}/{MAX_TRAIN_STEPS} episode={episode} "
                    f"episode_steps={episode_steps} q_states={len(agent.Q)}"
                )
                
    # ====================
    # Save Dataset
    # ====================
    print_stage("Save Dataset")
    try:
        minari.delete_dataset(DATASET_ID)
    except Exception:
        pass

    collector.create_dataset(
        dataset_id=DATASET_ID,
        algorithm_name="q_learning",
        author="local_test",
        author_email="local_test@example.com",
        code_permalink="https://example.com/test_agent_q_table",
        eval_env=eval_env,
        description="QTable training rollout recorded via Minari DataCollector",
    )
    state_path = state_collector.save(DATASET_ID)
    value_path = value_collector.save(DATASET_ID)

    # ====================
    # Verify Dataset
    # ====================
    print_stage("Verify Dataset")
    dataset = minari.load_dataset(DATASET_ID)
    print(f"dataset_id={dataset.spec.dataset_id}")
    print(f"episodes={dataset.total_episodes} total_steps={dataset.total_steps}")
    print(f"train_steps={total_steps} q_states={len(agent.Q)}")
    print(f"state_data_path={state_path}")
    print(f"value_data_path={value_path}")
finally:
    eval_env.close()
    collector.close()

