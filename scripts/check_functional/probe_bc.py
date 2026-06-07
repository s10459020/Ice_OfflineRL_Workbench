import gymnasium as gym
import numpy as np
import torch

from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.config.paths import data_path_test
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.dataset.loader.minari.collector import MinariCollectorWrapper
from ice_offline.store.probe.hopper_ood_action import HopperOodActionProbe
from ice_offline.store.probe.op_collector import ProbeCollectWrapper
from ice_offline.store.state.hopper import HopperState, HopperStateIO
from ice_offline.store.state.op_collector import StateCollectWrapper
from ice_offline.tools.printer import print_stage


MODEL_STEP = 200_000
EPISODES = 10
SEED = 42
PRINT_INTERVAL = 1
AGENT_ID = "bc_deterministic"
PROBE_SAMPLE_COUNT = 100


def make_eval_fn(agent: BCDeterministicAgent):
    def eval_fn(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        actor_actions = agent.act_batch(observations)
        errors = np.square(actor_actions - actions).sum(axis=1)
        return -errors.astype(np.float32)

    return eval_fn


def test(
    dataset: Dataset,
    *,
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    eval_env: gym.Env | None = None,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
) -> tuple[list[float], BCDeterministicAgent]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env = eval_env or dataset.make_env()

    print_stage("Probe BC Deterministic")
    agent = BCDeterministicAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
    )
    agent.load(dataset.id, model_step)

    returns = []
    for episode in range(1, episodes + 1):
        rewards = []
        observation, _ = eval_env.reset(seed=seed + episode)
        while True:
            action = agent.act(observation)
            observation, reward, terminated, truncated, _ = eval_env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        total_reward = float(sum(rewards))
        returns.append(total_reward)
        if print_interval and episode % print_interval == 0:
            print(f"Episode {episode}/{episodes}, Return: {total_reward:.2f}")

    return returns, agent


def collect(
    dataset: Dataset,
    *,
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
):
    agent = BCDeterministicAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim)
    agent.load(dataset.id, model_step)

    env = dataset.make_env()
    probe_col = ProbeCollectWrapper(
        env,
        HopperOodActionProbe(PROBE_SAMPLE_COUNT),
        make_eval_fn(agent),
    )
    state_col = StateCollectWrapper(probe_col, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    returns, _ = test(
        dataset=dataset,
        episodes=episodes,
        model_step=model_step,
        eval_env=minari_col,
        seed=seed,
        print_interval=print_interval,
    )

    data_path = data_path_test(dataset.id, AGENT_ID)
    minari_data = minari_col.save(data_path, id=dataset.id, agent_id=AGENT_ID)
    state_data = state_col.save(data_path)
    probe_data = probe_col.save(data_path)
    minari_col.close()

    return returns, minari_data, state_data, probe_data


if __name__ == "__main__":
    dataset = HopperSimpleDataset()
    returns, minari_data, state_data, probe_data = collect(
        dataset=dataset,
        episodes=EPISODES,
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"avg_returns={sum(returns) / len(returns):.2f}")
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
    print(f"state_data={state_data.path}")
    print(f"probe_data={probe_data.path}")
