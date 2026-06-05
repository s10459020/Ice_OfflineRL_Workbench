import numpy as np
import torch
import gymnasium as gym

from ice_offline.agent._spec import model_ref
from ice_offline.agent.scas_mean import ScasMeanAgent, ScasDynamic
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.tools.paths import dataset_root
from ice_offline.tools.printer import print_stage
from ice_offline.dataset.loader.minari.collector import MinariCollectorWrapper
from ice_offline.store.state.hopper import HopperState, HopperStateIO
from ice_offline.store.state.op_collector import StateCollectWrapper


DYNAMIC_STEP = 100_000
AGENT_MODEL_STEP = 200_000
EPISODES = 10
SEED = 42
PRINT_INTERVAL = 1


def test(
    dataset: Dataset,
    *,
    task_id: str = None,
    episodes: int = EPISODES,
    agent_model_step: int = AGENT_MODEL_STEP,
    dynamic_step: int = DYNAMIC_STEP,
    eval_env: gym.Env | None = None,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
) -> list[float]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_id = task_id or f"{dataset.id}-scas_mean-v0"
    eval_env = eval_env or dataset.make_env()

    print_stage("Test SCAS Mean")
    dynamics = ScasDynamic(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
    )
    dynamics.load(model_ref(f"{task_id}/dynamics", dynamic_step))

    agent = ScasMeanAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        dynamics=dynamics,
    )
    agent.load(model_ref(task_id, agent_model_step))

    returns = []
    for episode in range(1, episodes + 1):
        rewards = []
        o, _ = eval_env.reset(seed=seed + episode)
        while True:
            a = agent.act(o)
            o, r, terminated, truncated, _ = eval_env.step(a)
            rewards.append(r)

            if terminated or truncated:
                break
        total_reward = sum(rewards)
        returns.append(total_reward)

        if print_interval and episode % print_interval == 0:
            print(f"Episode {episode}/{episodes}, Return: {total_reward:.2f}")

    return returns


def collect(
    dataset: Dataset,
    *,
    task_id: str = None,
    episodes: int = EPISODES,
    model_step: int = AGENT_MODEL_STEP,
    dynamic_step: int = DYNAMIC_STEP,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
):
    task_id = task_id or f"{dataset.id}-scas_mean-v0"
    env = dataset.make_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    returns = test(
        dataset=dataset,
        task_id=task_id,
        episodes=episodes,
        agent_model_step=model_step,
        dynamic_step=dynamic_step,
        eval_env=minari_col,
        seed=seed,
        print_interval=print_interval,
    )

    minari_data = minari_col.save(f"test/{task_id}")
    state_data = state_col.save(dataset_root() / "test" / task_id / "data" / "main_data.hdf5")
    minari_col.close()

    return returns, minari_data, state_data


if __name__ == "__main__":
    dataset = HopperSimpleDataset()
    returns, minari_data, state_data = collect(
        dataset=dataset,
        task_id=f"{dataset.id}-scas_mean-v0",
        episodes=EPISODES,
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"total_returns={sum(returns)}")
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")


