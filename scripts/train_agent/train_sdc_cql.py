import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent._spec import agent_batch
from ice_offline.agent.sdc_cql import SDCCQLAgent
from ice_offline.dataset._spec import Dataset, TorchBuffer
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.data.minari.collector import MinariCollectorWrapper
from ice_offline.data.state.hopper import HopperState
from ice_offline.data.state.hopper import HopperStateIO
from ice_offline.data.state.op_collector import StateCollectWrapper
from ice_offline.data.state.op_dataset import StateDataset
from ice_offline.run.evaluator import Evaluator
from ice_offline.tools.printer import print_stage


BATCH_SIZE = 256
STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3
SAVE_INTERVAL = 20_000
SEED = 42
DEVICE = "cuda:0"


def eval_loss(agent: SDCCQLAgent, batch: TorchBuffer) -> dict[str, float]:
    batch = agent_batch(batch)
    with torch.no_grad():
        loss_td = agent.loss_td(batch)
        loss_suppress = agent.loss_suppress(batch)
        loss_critic = agent.loss_critic(batch)
        loss_actor = agent.loss_actor(batch)
        loss_dynamics = agent.loss_dynamics(batch)
        loss_transition = agent.loss_transition(batch)
        loss_sdc = agent.loss_state_deviation(batch[0])
        return {
            "loss_q_td": float(loss_td.item()),
            "loss_q_suppress": float(loss_suppress.sum().item()),
            "loss_actor": float(loss_actor.item()),
            "loss_critic": float(loss_critic.item()),
            "loss_dynamics": float(loss_dynamics.item()),
            "loss_transition": float(loss_transition.item()),
            "loss_sdc": float(loss_sdc.item()),
        }


def eval_return(batch: TorchBuffer) -> dict[str, float]:
    return {"return": float(batch.rew_list.sum().item())}


def train(
    dataset: Dataset,
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
    device: str = DEVICE,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_id = task_id or f"{dataset.id}-sdc_cql-v0"
    eval_env = eval_env or dataset.make_env()
    dataset.set_seed(seed)

    print_stage("Train SDC CQL")
    agent = SDCCQLAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
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
        batch = dataset.sample_batch(batch_size)
        agent.update(batch)
        evaluator.eval(step=step, agent=agent, batch_loader=dataset, batch_size=batch_size, eval_env=eval_env)
        evaluator.print(step)
        evaluator.recode(step)
        if step % save_interval == 0 or step == steps:
            agent.save(task_id, step)


def collect(
    dataset: Dataset,
    *,
    steps: int = STEPS,
    task_id: str = None,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_offline_n: int = EVAL_OFFLINE_N,
    eval_online_n: int = EVAL_ONLINE_N,
    save_interval: int = SAVE_INTERVAL,
    device: str = DEVICE,
) -> tuple[minari.MinariDataset, StateDataset]:
    task_id = task_id or f"{dataset.id}-sdc_cql-v0"
    env = dataset.make_env()
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
        device=device,
    )

    minari_data = minari_col.save(f"train/{task_id}")
    state_data = state_col.save(f"train/{task_id}")
    minari_col.close()

    return minari_data, state_data


if __name__ == "__main__":
    dataset = HopperSimpleDataset(device=DEVICE).load()
    minari_data, state_data = collect(dataset=dataset)
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
