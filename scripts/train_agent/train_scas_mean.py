import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.scas_mean import ScasMeanAgent
from ice_offline.agent.scas_mean import ScasDynamic
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset._types import Batch
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.dataset.loader.minari.collector import MinariCollectorWrapper
from ice_offline.store.state.hopper import HopperState
from ice_offline.store.state.hopper import HopperStateIO
from ice_offline.store.state.op_collector import StateCollectWrapper
from ice_offline.store.state.op_dataset import StateDataset
from ice_offline.run.evaluator import Evaluator
from ice_offline.tools.paths import dataset_root
from ice_offline.tools.printer import print_stage


BATCH_SIZE = 256
DYNAMIC_STEPS = 100_000
AGENT_STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3
SAVE_INTERVAL = 20_000
SEED = 42
DEVICE = "cuda:0"


def eval_loss_dynamic(dynamics: ScasDynamic, batch: Batch) -> dict[str, float]:
    s = batch[0]
    a = batch[1]
    sn = batch[3]
    with torch.no_grad():
        return {"1. loss_dynamic": float(dynamics.loss_dynamic(s, a, sn).item())}


def eval_loss_agent(agent: ScasMeanAgent, batch: Batch) -> dict[str, float]:
    with torch.no_grad():
        loss_td3 = agent.loss_td3(batch)
        loss_correction = agent.loss_correction(batch)
        loss_critic = agent.loss_critic(batch)
        loss_actor = agent.loss_actor(batch)
        return {
            "2. loss_td3": float(loss_td3.item()),
            "3. loss_correction": float(loss_correction.item()),
            "4. loss_actor": float(loss_actor.item()),
            "5. loss_critic": float(loss_critic.item()),
        }


def eval_return(batch: Batch) -> dict[str, float]:
    return {"6. return": float(batch[2].sum().item())}


def train(
    dataset: Dataset,
    *,
    dynamic_steps: int = DYNAMIC_STEPS,
    agent_steps: int = AGENT_STEPS,
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

    task_id = task_id or f"{dataset.env_id}_scas_mean-v0"
    eval_env = eval_env or dataset.make_env()
    dataset.set_seed(seed)

    print_stage("Train SCAS Mean Dynamics")
    dynamics = ScasDynamic(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
    )
    dynamics_evaluator = Evaluator(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_offline_fns=[eval_loss_dynamic],
    )
    for step in range(1, dynamic_steps + 1):
        batch = dataset.sample_batch(batch_size)
        dynamics.update(batch)
        dynamics_evaluator.eval(step=step, agent=dynamics, batch_loader=dataset, batch_size=batch_size)
        dynamics_evaluator.print(step)
        dynamics_evaluator.recode(step)
        if step % save_interval == 0 or step == dynamic_steps:
            dynamics.save(f"{task_id}/dynamics", step)

    print_stage("Train SCAS Mean Agent")
    agent = ScasMeanAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        dynamics=dynamics,
        device=device,
    )
    agent_evaluator = Evaluator(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss_agent],
        eval_online_fns=[eval_return],
        recode_reset=False,
    )
    for step in range(1, agent_steps + 1):
        batch = dataset.sample_batch(batch_size)
        agent.update(batch)
        agent_evaluator.eval(step=step, agent=agent, batch_loader=dataset, batch_size=batch_size, eval_env=eval_env)
        agent_evaluator.print(step)
        agent_evaluator.recode(step)
        if step % save_interval == 0 or step == agent_steps:
            agent.save(task_id, step)


def collect(
    dataset: Dataset,
    *,
    steps: int = AGENT_STEPS,
    dynamic_step: int = DYNAMIC_STEPS,
    task_id: str = None,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_offline_n: int = EVAL_OFFLINE_N,
    eval_online_n: int = EVAL_ONLINE_N,
    save_interval: int = SAVE_INTERVAL,
    device: str = DEVICE,
) -> tuple[minari.MinariDataset, StateDataset]:
    task_id = task_id or f"{dataset.env_id}_scas_mean-v0"
    env = dataset.make_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    train(
        task_id=task_id,
        dataset=dataset,
        eval_env=minari_col,
        batch_size=batch_size,
        dynamic_steps=dynamic_step,
        agent_steps=steps,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        save_interval=save_interval,
        device=device,
    )

    minari_data = minari_col.save(f"train/{task_id}")
    state_data = state_col.save(dataset_root() / "train" / task_id / "data" / "main_data.hdf5")
    minari_col.close()

    return minari_data, state_data


if __name__ == "__main__":
    dataset = HopperSimpleDataset(device=DEVICE)
    minari_data, state_data = collect(dataset=dataset)
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")


