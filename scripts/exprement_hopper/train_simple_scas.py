import gymnasium as gym
from ice_offline.dataset._spec import BaseDataset
import minari
import numpy as np
import torch

from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.pipeline.minari_collector import MinariCollectorWrapper
from ice_offline.pipeline.state.hopper import HopperState
from ice_offline.pipeline.state.hopper import HopperStateIO
from ice_offline.pipeline.state_operator.state_collector import StateCollectWrapper
from ice_offline.pipeline.state_operator.state_dataset import StateDataset
from ice_offline.runner.evaluator2 import Evaluator2
from ice_offline.tools.printer import print_stage


TASK_ID = "train/hopper_simple_scas-v0"

BATCH_SIZE = 256
DYNAMICS_STEPS = 100_000
AGENT_STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3
SAVE_INTERVAL = 20_000


def eval_loss_dynamic(dynamics: ScasDynamic, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    s, a, _, sn, _ = episode_batch
    with torch.no_grad():
        return {"loss_dynamic": float(dynamics.loss_dynamic(s, a, sn).item())}


def eval_loss_agent(agent: ScasAgent, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    s, a, r, sn, d = episode_batch
    with torch.no_grad():
        loss_q = agent.loss_critic(s, a, r, sn, d)
        loss_td3 = agent.loss_td3(s)
        loss_correction = agent.loss_correction(s, sn)
        loss_pi = agent.loss_actor(s, sn)
        return {
            "loss_q": float(loss_q.item()),
            "loss_td3": float(loss_td3.item()),
            "loss_correction": float(loss_correction.item()),
            "loss_pi": float(loss_pi.item()),
            "loss_pi_weighted": float(((1.0 - agent.lmbda) * loss_td3 + agent.lmbda * loss_correction).item()),
        }


def eval_reward(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"reward_sum": float(r.sum().item())}


def train(
    task_id: str,
    dataset: BaseDataset,
    *,
    eval_env: gym.Env,
    seed: int = 42,
    batch_size: int = 256,
    dynamics_steps: int = 100_000,
    agent_steps: int = 200_000,
    eval_interval: int = 2_000,
    eval_offline_n: int = 8,
    eval_online_n: int = 3,
    save_interval: int = 20_000,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_loader = MinariLoader(dataset=dataset, seed=seed)

    print_stage("Train SCAS Dynamics")
    dynamics = ScasDynamic(
        obs_dim=batch_loader.obs_dim,
        act_dim=batch_loader.act_dim,
        learning_rate=1e-3,
        device="cpu",
    )
    dynamics_evaluator = Evaluator2(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_offline_fns=[eval_loss_dynamic],
    )
    for step in range(1, dynamics_steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        dynamics.update(batch)
        dynamics_evaluator.eval_offline(step=step, agent=dynamics, batch_loader=batch_loader, batch_size=batch_size)
        dynamics_evaluator.print(step)
        dynamics_evaluator.recode(step)
        if step % save_interval == 0 or step == dynamics_steps:
            dynamics.save(f"{task_id}/dynamics", step)

    print_stage("Train SCAS Agent")
    agent = ScasAgent(
        obs_dim=batch_loader.obs_dim,
        act_dim=batch_loader.act_dim,
        dynamics=dynamics,
        max_action=1.0,
        device="cpu",
    )
    agent_evaluator = Evaluator2(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss_agent],
        eval_online_fns=[eval_reward],
        recode_reset=False,
    )
    for step in range(1, agent_steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)
        agent_evaluator.eval_offline(step=step, agent=agent, batch_loader=batch_loader, batch_size=batch_size)
        agent_evaluator.eval_online(step=step, agent=agent, env=eval_env)
        agent_evaluator.print(step)
        agent_evaluator.recode(step)
        if step % save_interval == 0 or step == agent_steps:
            agent.save(task_id, step)


def collect(
    env_id: str,
    task_id: str = TASK_ID,
    dataset: BaseDataset,
    batch_size: int = BATCH_SIZE,
    dynamics_steps: int = DYNAMICS_STEPS,
    agent_steps: int = AGENT_STEPS,
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
        dynamics_steps=dynamics_steps,
        agent_steps=agent_steps,
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


