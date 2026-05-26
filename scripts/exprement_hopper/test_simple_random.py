import gymnasium as gym
import minari

from ice_offline.pipeline.minari.collector import MinariCollectorWrapper
from ice_offline.pipeline.state.hopper import HopperState, HopperStateIO
from ice_offline.pipeline.state.op_collector import StateCollectWrapper
from ice_offline.pipeline.state.op_dataset import StateDataset
from ice_offline.tools.printer import print_stage


TASK_ID = "test/test_simple_random-v0"
ENV_ID = "Hopper-v5"
TEST_EPISODES = 1000
PRINT_INTERVAL = 10


def test(
    env: gym.Env,
    action_space,
    test_episodes: int = TEST_EPISODES,
    print_interval: int = PRINT_INTERVAL,
) -> None:
    print_stage("Test Random")
    for episode in range(1, test_episodes + 1):
        _, _ = env.reset()
        done = False
        while not done:
            act = action_space.sample()
            _, _, terminated, truncated, _ = env.step(act)
            done = bool(terminated or truncated)
        if episode % print_interval == 0 or episode == test_episodes:
            print(f"{episode}/{test_episodes}")


def collect(
    env_id: str = ENV_ID,
    task_id: str = TASK_ID,
    test_episodes: int = TEST_EPISODES,
    print_interval: int = PRINT_INTERVAL,
) -> tuple[minari.MinariDataset, StateDataset]:
    env = gym.make(env_id)
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    test(env=minari_col, action_space=env.action_space, test_episodes=test_episodes, print_interval=print_interval)

    minari_data = minari_col.save(task_id)
    state_data = state_col.save(task_id)
    minari_col.close()
    return minari_data, state_data


if __name__ == "__main__":
    minari_data, state_data = collect(
        env_id=ENV_ID,
        task_id=TASK_ID,
        test_episodes=TEST_EPISODES,
        print_interval=PRINT_INTERVAL,
    )

    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")



