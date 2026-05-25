import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.cql_continuous import CQLAgentContinuous
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.pipeline.minari_collector import MinariCollectorWrapper
from ice_offline.pipeline.state.hopper import HopperState
from ice_offline.pipeline.state.hopper import HopperStateIO
from ice_offline.pipeline.state_operator.state_collector import StateCollectWrapper
from ice_offline.pipeline.state_operator.state_dataset import StateDataset
from ice_offline.runner.evaluator2 import Evaluator2
from ice_offline.tools.printer import print_stage


TASK_ID = "train/hopper_simple_cql-v0"

BATCH_SIZE = 256
STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3
SAVE_INTERVAL = 20_000


def eval_loss(agent: CQLAgentContinuous, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, r, on, d = episode_batch
    with torch.no_grad():
        loss_td_parts = agent._loss_td(o, a, r, on, d)
        loss_cql_parts = agent._loss_cql(o, a, on)
        loss_critic = agent.loss_critic(o, a, r, on, d)
        loss_pi = agent.loss_actor(o)
        return {
            "loss_q": float(loss_critic.item()),
            "loss_q_td1": float(loss_td_parts[0].item()),
            "loss_q_td2": float(loss_td_parts[1].item()),
            "loss_q_cql1": float(loss_cql_parts[0].item()),
            "loss_q_cql2": float(loss_cql_parts[1].item()),
            "loss_pi": float(loss_pi.item()),
        }


def eval_return(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"return": float(r.sum().item())}


def train(
    task_id: str,
    dataset,
    *,
    eval_env: gym.Env,
    seed: int = 42,
    batch_size: int = 256,
    steps: int = 200_000,
    eval_interval: int = 2_000,
    eval_offline_n: int = 8,
    eval_online_n: int = 3,
    save_interval: int = 20_000,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_loader = MinariLoader(dataset=dataset, seed=seed)

    print_stage("Train CQL")
    agent = CQLAgentContinuous(
        obs_size=batch_loader.obs_dim,
        act_size=batch_loader.act_dim,
    )

    evaluator = Evaluator2(
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
        evaluator.eval_offline(step=step, agent=agent, batch_loader=batch_loader, batch_size=batch_size)
        evaluator.eval_online(step=step, agent=agent, env=eval_env)
        evaluator.print(step)
        evaluator.recode(step)
        if step % save_interval == 0 or step == steps:
            agent.save(task_id, step)


def collect(
    dataset,
    env_id: str = "Hopper-v5",
    task_id: str = TASK_ID,
    batch_size: int = BATCH_SIZE,
    steps: int = STEPS,
    eval_interval: int = EVAL_INTERVAL,
    eval_offline_n: int = EVAL_OFFLINE_N,
    eval_online_n: int = EVAL_ONLINE_N,
    save_interval: int = SAVE_INTERVAL,
) -> tuple[minari.MinariDataset, StateDataset]:
    env = gym.make(env_id)
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

    minari_data = minari_col.save(task_id)
    state_data = state_col.save(task_id)
    minari_col.close()

    return minari_data, state_data


if __name__ == "__main__":
    dataset = get_dataset("hopper_simple")
    minari_data, state_data = collect(dataset=dataset, env_id=dataset.env_id, task_id=TASK_ID)
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")


