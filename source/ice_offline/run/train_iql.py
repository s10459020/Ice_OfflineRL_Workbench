import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.iql_continuous import IQLAgentContinuous
from ice_offline.dataset._lookup import get_dataset
from ice_offline.dataset._spec import BaseDataset
from ice_offline.pipeline.minari.loader import MinariLoader
from ice_offline.pipeline.minari.collector import MinariCollectorWrapper
from ice_offline.pipeline.state.hopper import HopperState
from ice_offline.pipeline.state.hopper import HopperStateIO
from ice_offline.pipeline.state.op_collector import StateCollectWrapper
from ice_offline.pipeline.state.op_dataset import StateDataset
from ice_offline.run.evaluator import Evaluator
from ice_offline.tools.printer import print_stage


DATASET_KEY = "hopper_simple"
BATCH_SIZE = 256
STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3
SAVE_INTERVAL = 20_000
SEED = 42


def eval_loss(agent: IQLAgentContinuous, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, r, on, d = episode_batch
    with torch.no_grad():
        return {
            "loss_q": float(agent._loss_q(o, a, r, on, d).item()),
            "loss_critic": float(agent.loss_critic(o, a, r, on, d).item()),
            "loss_v": float(agent._loss_v(o, a).item()),
            "loss_pi": float(agent.loss_actor(o, a).item()),
        }


def eval_return(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"return": float(r.sum().item())}


def train(
    dataset: BaseDataset,
    *,
    steps: int = STEPS,
    task_id: str = None,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_offline_n: int = EVAL_OFFLINE_N,
    eval_online_n: int = EVAL_ONLINE_N,
    eval_env: gym.Env | None = None,
    save_interval: int = SAVE_INTERVAL,
    seed: int = SEED,
) -> None:
    task_id = task_id or f"{dataset.dataset_id.replace('/', '_')}_iql-v0"
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_loader = MinariLoader(dataset=dataset, seed=seed)
    eval_env = eval_env or dataset.make_collect_env()

    print_stage("Train IQL")
    agent = IQLAgentContinuous(
        obs_size=batch_loader.obs_dim,
        act_size=batch_loader.act_dim,
    )

    evaluator = Evaluator(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss],
        eval_online_fns=[eval_return],
    )

    for step in range(1, steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)
        evaluator.eval(step=step, agent=agent, batch_loader=batch_loader, batch_size=batch_size, eval_env=eval_env)
        evaluator.print(step)
        evaluator.recode(step)
        if step % save_interval == 0 or step == steps:
            agent.save(task_id, step)


def collect(
    dataset: BaseDataset,
    *,
    steps: int = STEPS,
    task_id: str = None,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_offline_n: int = EVAL_OFFLINE_N,
    eval_online_n: int = EVAL_ONLINE_N,
    save_interval: int = SAVE_INTERVAL,
) -> tuple[minari.MinariDataset, StateDataset]:
    task_id = task_id or f"{dataset.dataset_id.replace('/', '_')}_iql-v0"
    env = dataset.make_collect_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    train(
        task_id=task_id,
        dataset=dataset,
        eval_env=minari_col,
        batch_size=batch_size,
        steps=steps,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        save_interval=save_interval,
    )

    minari_data = minari_col.save(f"train/{task_id}")
    state_data = state_col.save(f"train/{task_id}")
    minari_col.close()

    return minari_data, state_data


if __name__ == "__main__":
    dataset = get_dataset(DATASET_KEY)
    minari_data, state_data = collect(dataset=dataset)
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")






