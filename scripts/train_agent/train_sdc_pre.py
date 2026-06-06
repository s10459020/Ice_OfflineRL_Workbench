import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.sdc_pre import SDCPreAgent
from ice_offline.agent.sdc_pre import SDCPreModel
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
MODEL_STEPS = 100_000
AGENT_STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3
SAVE_INTERVAL = 20_000
SEED = 42
DEVICE = "cuda:0"
AGENT_ID = "sdc_pre"


def eval_loss_model(model: SDCPreModel, batch: Batch) -> dict[str, float]:
    with torch.no_grad():
        loss_dynamics = model.loss_dynamics(batch)
        loss_transition = model.loss_transition(batch)
        loss_state_models = model.loss_state_models(batch)
        return {
            "1. loss_dynamics": float(loss_dynamics.item()),
            "2. loss_transition": float(loss_transition.item()),
            "3. loss_state_models": float(loss_state_models.item()),
        }


def eval_loss_agent(agent: SDCPreAgent, batch: Batch) -> dict[str, float]:
    with torch.no_grad():
        loss_td = agent.loss_td(batch)
        loss_suppress = agent.loss_suppress(batch)
        loss_critic = agent.loss_critic(batch)
        loss_actor = agent.loss_actor(batch)
        loss_sdc = agent.loss_state_deviation(batch[0])
        return {
            "4. loss_sdc": float(loss_sdc.item()),
            "5. loss_actor": float(loss_actor.item()),
            "6. loss_td": float(loss_td.item()),
            "7. loss_suppress": float(loss_suppress.sum().item()),
            "8. loss_critic": float(loss_critic.item()),
        }


def eval_return(batch: Batch) -> dict[str, float]:
    return {"9. return": float(batch[2].sum().item())}


def train(
    dataset: Dataset,
    *,
    model_steps: int = MODEL_STEPS,
    agent_steps: int = AGENT_STEPS,
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

    print_stage("Train SDC Pre State Models")
    state_models = SDCPreModel(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
    )
    model_evaluator = Evaluator(
        dataset_id=dataset.id,
        agent_id=AGENT_ID,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_offline_fns=[eval_loss_model],
    )
    for step in range(1, model_steps + 1):
        batch = dataset.sample_batch(batch_size)
        state_models.update(batch)
        model_evaluator.eval(step=step, agent=state_models, batch_loader=dataset, batch_size=batch_size)
        model_evaluator.print(step)
        model_evaluator.recode(step)
        if step % save_interval == 0 or step == model_steps:
            state_models.save(dataset.id, step)

    print_stage("Train SDC Pre Agent")
    agent = SDCPreAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        state_models=state_models,
        device=device,
    )
    agent_evaluator = Evaluator(
        dataset_id=dataset.id,
        agent_id=AGENT_ID,
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
            agent.save(dataset.id, step)


def collect(
    dataset: Dataset,
    *,
    steps: int = AGENT_STEPS,
    steps_model: int = MODEL_STEPS,
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
        model_steps=steps_model,
        agent_steps=steps,
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
    dataset = HopperSimpleDataset()
    minari_data, state_data = collect(dataset)
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
    print(f"state_data={state_data}")




