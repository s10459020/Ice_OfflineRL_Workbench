import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset._types import Batch
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.dataset.loader.minari.collector import MinariCollectorWrapper
from ice_offline.store.state.hopper import HopperState
from ice_offline.store.state.hopper import HopperStateIO
from ice_offline.store.state.op_collector import StateCollectWrapper
from ice_offline.store.state.op_dataset import StateDataset
from ice_offline.run.evaluator import Evaluator
from ice_offline.config.paths import data_path_train
from ice_offline.tools.printer import print_stage


BATCH_SIZE = 256
STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3
SAVE_INTERVAL = 20_000
SEED = 42
DEVICE = "cuda:0"
AGENT_ID = "td3bc"


def eval_loss(agent: TD3BCAgent, batch: Batch) -> dict[str, float]:
    with torch.no_grad():
        return {
            "1. loss_td3": float(agent.loss_td3(batch).item()),
            "2. loss_bc": float(agent.loss_bc(batch).item()),
            "3. loss_actor": float(agent.loss_actor(batch).item()),
            "4. loss_critic": float(agent.loss_critic(batch).item()),
        }


def eval_return(batch: Batch) -> dict[str, float]:
    return {"5. return": float(batch[2].sum().item())}


def train(
    dataset: Dataset,
    *,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_offline_n: int = EVAL_OFFLINE_N,
    eval_online_n: int = EVAL_ONLINE_N,
    eval_env: gym.Env | None = None,
    save_interval: int = SAVE_INTERVAL,
    seed: int = SEED,
    device: str = DEVICE,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env = eval_env or dataset.make_env()
    dataset.set_seed(seed)

    print_stage("Train TD3BC")
    agent = TD3BCAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
    )

    evaluator = Evaluator(
        dataset_id=dataset.id,
        agent_id=AGENT_ID,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss],
        eval_online_fns=[eval_return],
    )

    for step in range(1, steps + 1):
        batch = dataset.sample_batch(batch_size)
        agent.update(batch)
        evaluator.eval(step=step, agent=agent, batch_loader=dataset, batch_size=batch_size, eval_env=eval_env)
        evaluator.print(step)
        evaluator.recode(step)
        if step % save_interval == 0 or step == steps:
            agent.save(dataset.id, step)


def collect(
    dataset: Dataset,
    *,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_offline_n: int = EVAL_OFFLINE_N,
    eval_online_n: int = EVAL_ONLINE_N,
    save_interval: int = SAVE_INTERVAL,
    device: str = DEVICE,
) -> tuple[minari.MinariDataset, StateDataset]:
    env = dataset.make_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    train(
        dataset=dataset,
        eval_env=minari_col,
        batch_size=batch_size,
        steps=steps,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        save_interval=save_interval,
        device=device,
    )

    data_path = data_path_train(dataset.id, AGENT_ID)
    minari_data = minari_col.save(data_path)
    state_data = state_col.save(data_path)
    minari_col.close()

    return minari_data, state_data


if __name__ == "__main__":
    dataset = HopperSimpleDataset(device=DEVICE)
    minari_data, state_data = collect(dataset=dataset)
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")




