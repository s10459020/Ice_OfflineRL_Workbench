import gymnasium as gym
import minari
import numpy as np

from ice_offline.agent._spec import model_ref
from ice_offline.agent.bc_continuous_deterministic import BCAgentContinuousDeterministic
from ice_offline.pipeline.minari.collector import MinariCollectorWrapper
from ice_offline.pipeline.state.hopper import HopperState, HopperStateIO
from ice_offline.pipeline.state.op_collector import StateCollectWrapper
from ice_offline.pipeline.state.op_dataset import StateDataset
from ice_offline.tools.printer import print_stage


TASK_ID = "test/test_simple_bc-v0"
ENV_ID = "Hopper-v5"
TEST_EPISODES = 1000
MODEL_TASK_ID = "train/hopper_simple_bc-v0"
MODEL_STEP = 200_000


def test(env: gym.Env, test_episodes: int, model_task_id: str, model_step: int) -> None:
    print_stage("Test BC")
    obs_size = int(np.prod(env.observation_space.shape))
    act_size = int(np.prod(env.action_space.shape))
    agent = BCAgentContinuousDeterministic(obs_size=obs_size, act_size=act_size)
    agent.load(model_ref(model_task_id, model_step))

    for _ in range(test_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            act = agent.act_best(obs)
            obs, _, terminated, truncated, _ = env.step(act)
            done = bool(terminated or truncated)


def collect(
    env_id: str = ENV_ID,
    task_id: str = TASK_ID,
    model_task_id: str = MODEL_TASK_ID,
    test_episodes: int = TEST_EPISODES,
    model_step: int = MODEL_STEP,
) -> tuple[minari.MinariDataset, StateDataset]:
    env = gym.make(env_id)
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    test(
        env=minari_col,
        test_episodes=test_episodes,
        model_task_id=model_task_id,
        model_step=model_step,
    )

    minari_data = minari_col.save(task_id)
    state_data = state_col.save(task_id)
    minari_col.close()
    return minari_data, state_data


if __name__ == "__main__":
    minari_data, state_data = collect(
        env_id=ENV_ID,
        task_id=TASK_ID,
        test_episodes=TEST_EPISODES,
        model_task_id=MODEL_TASK_ID,
        model_step=MODEL_STEP,
    )

    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")




