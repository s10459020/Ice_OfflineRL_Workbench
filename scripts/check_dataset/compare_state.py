import gymnasium as gym
import minari
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.data import State
from ice_offline.dataset import StateCollector, StateLoader
from ice_offline.env.common import MissionTextWrapper, NoJpegImageWrapper
from ice_offline.tools import print_stage


# ====================
# Config
# ====================
ENV_ID = "BabyAI-OneRoomS8-v0"
DATASET_ID = "check/recode_state-v0"
MAX_EPISODES = 5
SEED_BASE = 42


def _make_env() -> gym.Env:
    env = FullyObsWrapper(gym.make(ENV_ID))
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    return env


def _assert_state_equal(expected: State, actual: State, *, episode: int, step: int) -> None:
    if expected.mission != actual.mission:
        raise SystemExit(f"FAIL: mission mismatch at episode={episode} step={step}")
    if expected.agent_pos != actual.agent_pos:
        raise SystemExit(f"FAIL: agent_pos mismatch at episode={episode} step={step}")
    if expected.agent_dir != actual.agent_dir:
        raise SystemExit(f"FAIL: agent_dir mismatch at episode={episode} step={step}")
    if expected.carrying != actual.carrying:
        raise SystemExit(f"FAIL: carrying mismatch at episode={episode} step={step}")
    if not np.array_equal(expected.grid, actual.grid):
        raise SystemExit(f"FAIL: grid mismatch at episode={episode} step={step}")


# ====================
# Collect
# ====================
print_stage("Collect")
eval_env = _make_env()
base_env = _make_env()
state_collector = StateCollector(base_env)
collector = minari.DataCollector(state_collector, record_infos=False)

expected_episodes: list[list[State]] = []
total_steps = 0

try:
    for episode in range(1, MAX_EPISODES + 1):
        _, _ = collector.reset(seed=SEED_BASE + episode)
        expected_steps: list[State] = [state_collector.get_last()]

        done = False
        truncated = False
        episode_steps = 0
        while not (done or truncated):
            action = int(collector.action_space.sample())
            _, _, done, truncated, _ = collector.step(action)
            expected_steps.append(state_collector.get_last())
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
        algorithm_name="random_policy",
        author="local_test",
        author_email="local_test@example.com",
        code_permalink="https://example.com/check_compare_state",
        eval_env=eval_env,
        description="collect and compare state_data with loader",
    )
    state_data_path = state_collector.save(DATASET_ID)
    print(f"state_data_path={state_data_path}")
finally:
    eval_env.close()
    collector.close()

print(f"total_steps={total_steps}")


# ====================
# Compare
# ====================
print_stage("Compare")
loader = StateLoader(DATASET_ID)
try:
    loaded_episode_count = loader.get_episode_count()
    expected_episode_count = len(expected_episodes)
    if loaded_episode_count != expected_episode_count:
        raise SystemExit(
            f"FAIL: episode_count mismatch expected={expected_episode_count} actual={loaded_episode_count}"
        )

    for ep_index, expected_steps in enumerate(expected_episodes):
        actual_steps = loader.load_episode(ep_index)
        if len(actual_steps) != len(expected_steps):
            raise SystemExit(
                f"FAIL: step_count mismatch at episode={ep_index} expected={len(expected_steps)} actual={len(actual_steps)}"
            )
        for step_index, (expected_state, actual_state) in enumerate(zip(expected_steps, actual_steps)):
            _assert_state_equal(expected_state, actual_state, episode=ep_index, step=step_index)
finally:
    loader.close()

print("PASS: compare_state all matched")
