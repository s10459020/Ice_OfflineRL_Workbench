import gymnasium as gym
import numpy as np
import torch

from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.config.paths import data_path_probe
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.store.minari.collector import MinariCollectorWrapper
from ice_offline.store.probe.action_axis_probe import ActionAxisProbe
from ice_offline.store.probe.op_collector import ProbeCollectWrapper
from ice_offline.store.state.hopper import HopperState, HopperStateIO
from ice_offline.store.state.op_collector import StateCollectWrapper
from ice_offline.tools.printer import print_stage


MODEL_STEP = 10_000
EPISODES = 2
SEED = 42
PRINT_INTERVAL = 1
AGENT_ID = "bc_deterministic"
PROBE_SAMPLE_COUNT = 100


def run(
    agent: BCDeterministicAgent,
    eval_env: gym.Env,
    *,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    print_stage("Probe BC Deterministic")
    for episode in range(1, episodes + 1):
        observation, _ = eval_env.reset(seed=seed)
        while True:
            action = agent.act(observation)
            observation, _, terminated, truncated, _ = eval_env.step(action)
            if terminated or truncated:
                break

        if print_interval and episode % print_interval == 0:
            print(f"Episode {episode}/{episodes}")


def probe(
    dataset: Dataset,
    *,
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
):
    agent = BCDeterministicAgent(obs_size=dataset.obs_dim, act_size=dataset.act_dim)
    agent.load(dataset.id, model_step)

    def eval_fn(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        actor_actions = agent.act_batch(observations)
        errors = np.square(actor_actions - actions).sum(axis=1)
        return -errors.astype(np.float32)

    env = dataset.make_env()
    probe_col = ProbeCollectWrapper(env, ActionAxisProbe(PROBE_SAMPLE_COUNT), eval_fn)
    state_col = StateCollectWrapper(probe_col, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    run(
        agent=agent,
        eval_env=minari_col,
        episodes=episodes,
        seed=seed,
        print_interval=print_interval,
    )

    data_path = data_path_probe(dataset.id, AGENT_ID)
    minari_data = minari_col.save(data_path, id=dataset.id, agent_id=AGENT_ID)
    state_data = state_col.save(data_path)
    probe_data = probe_col.save(data_path)
    minari_col.close()

    return minari_data, state_data, probe_data


if __name__ == "__main__":
    dataset = HopperSimpleDataset()
    minari_data, state_data, probe_data = probe(
        dataset=dataset,
        episodes=EPISODES,
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
    print(f"state_data={state_data.path}")
    print(f"probe_data={probe_data.path}")
