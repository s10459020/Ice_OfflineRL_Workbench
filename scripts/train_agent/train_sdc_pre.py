import math
import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.sdc_pre import SDCPreAgent
from ice_offline.agent.sdc_pre import SDCPreModel
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset._types import Batch
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.store.minari.collector import MinariCollectorWrapper
from ice_offline.store.state.hopper import HopperState
from ice_offline.store.state.hopper import HopperStateIO
from ice_offline.store.state.op_collector import StateCollectWrapper
from ice_offline.store.state.op_dataset import StateDataset
from ice_offline.store.eval.record import Evaluator
from ice_offline.store.metric.record import MetricRecorder
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.tools.printer import print_stage


STEPS = 200_000
MODEL_STEPS = math.ceil(STEPS/2)
SAVE_INTERVAL = math.ceil(STEPS/10)
EVAL_INTERVAL = math.ceil(STEPS/100)
PRINT_INTERVAL = math.ceil(STEPS/1000)

SEED = 42
BATCH_SIZE = 256
EVAL_EPISODES = 10
DEVICE = "cuda:0"
AGENT_ID = "sdc_pre"


def print_latest(step: int, recorder: MetricRecorder) -> None:
    metrics = recorder.last
    parts = [f"{name}={value:.6g}" for name, value in metrics.items()]
    print(f"train step={step}", *parts)


def update_model_with_record(recorder: MetricRecorder, model: SDCPreModel, batch: Batch) -> None:
    loss_dynamics = model.loss_dynamics(batch)
    loss_transition = model.loss_transition(batch)
    loss_state_models = loss_dynamics + loss_transition

    recorder.add("loss_dynamics", loss_dynamics)
    recorder.add_grad_norm("grad_dynamics", loss_dynamics, model.dynamics.parameters())
    recorder.add("loss_transition", loss_transition)
    recorder.add_grad_norm("grad_transition", loss_transition, model.transition.parameters())
    recorder.add("loss_state_models", loss_state_models)
    recorder.add_grad_norm(
        "grad_state_models",
        loss_state_models,
        list(model.dynamics.parameters()) + list(model.transition.parameters()),
    )
    recorder.flush()

    model.update(batch)


def update_agent_with_record(recorder: MetricRecorder, agent: SDCPreAgent, batch: Batch) -> None:
    o, _, _, _, _ = batch
    loss_sdc = agent.loss_state_deviation(batch)
    a_actor, _ = agent.actor.sample(o)
    q_actor = agent.critic.q_min(o, a_actor)
    loss_actor = -q_actor.mean() + agent.sdc_weight * (loss_sdc - agent.sdc_threshold)
    loss_td = agent.loss_td(batch)
    loss_suppress_raw = agent.loss_suppress(batch)
    loss_suppress = loss_suppress_raw.sum()
    loss_suppress_scaled = agent.critic.conservative_weight * (
        loss_suppress_raw - agent.critic.alpha_threshold
    )
    loss_multiplier = agent.multiplier.loss(loss_suppress_scaled.detach())
    loss_critic = loss_td + (agent.multiplier() * loss_suppress_scaled).sum()

    recorder.add("loss_sdc", loss_sdc)
    recorder.add_grad_norm("grad_sdc", loss_sdc, agent.actor.parameters())
    recorder.add("loss_actor", loss_actor)
    recorder.add_grad_norm("grad_actor", loss_actor, agent.actor.parameters())
    recorder.add("loss_td", loss_td)
    recorder.add_grad_norm("grad_td", loss_td, agent.critic.parameters())
    recorder.add("loss_suppress", loss_suppress)
    recorder.add_grad_norm("grad_suppress", loss_suppress, agent.critic.parameters())
    recorder.add("loss_multiplier", loss_multiplier)
    recorder.add_grad_norm("grad_multiplier", loss_multiplier, agent.multiplier.parameters())
    recorder.add("loss_critic", loss_critic)
    recorder.add_grad_norm("grad_critic", loss_critic, agent.critic.parameters())
    recorder.flush()

    agent.update(batch)


def train(
    dataset: Dataset,
    *,
    steps: int = STEPS,
    model_steps: int = MODEL_STEPS,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_episodes: int = EVAL_EPISODES,
    eval_env: gym.Env | None = None,
    save_interval: int = SAVE_INTERVAL,
    print_interval: int = PRINT_INTERVAL,
    seed: int = SEED,
    device: str = DEVICE,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env = eval_env or dataset.make_eval_env()
    dataset.set_seed(seed)

    print_stage("Train SDC Pre State Models")
    state_models = SDCPreModel(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
    )
    model_recorder = MetricRecorder(dataset.id, state_models.agent_name)
    for step in range(1, model_steps + 1):
        batch = dataset.sample_batch(batch_size)
        update_model_with_record(model_recorder, state_models, batch)
        if print_interval > 0 and step % print_interval == 0:
            print_latest(step, model_recorder)
        if step % save_interval == 0 or step == model_steps:
            state_models.save(dataset.id, step)

    print_stage("Train SDC Pre Agent")
    agent = SDCPreAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        state_models=state_models,
        device=device,
    )
    recorder = MetricRecorder(dataset.id, AGENT_ID)
    evaluator = Evaluator(dataset.id, AGENT_ID, episodes=eval_episodes)
    for step in range(1, steps + 1):
        batch = dataset.sample_batch(batch_size)
        update_agent_with_record(recorder, agent, batch)
        if print_interval > 0 and step % print_interval == 0:
            print_latest(step, recorder)
        if eval_interval > 0 and step % eval_interval == 0:
            avg_return = evaluator.eval(step, agent, eval_env)
            print(f"eval step={step} avg_return={avg_return:.6g}")
        if step % save_interval == 0 or step == steps:
            agent.save(dataset.id, step)


def collect(
    dataset: Dataset,
    *,
    steps: int = STEPS,
    steps_models: int = MODEL_STEPS,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_episodes: int = EVAL_EPISODES,
    save_interval: int = SAVE_INTERVAL,
    print_interval: int = PRINT_INTERVAL,
    device: str = DEVICE,
) -> tuple[minari.MinariDataset, StateDataset]:
    env = dataset.make_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    train(
        dataset=dataset,
        eval_env=minari_col,
        batch_size=batch_size,
        model_steps=steps_models,
        agent_steps=steps,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        save_interval=save_interval,
        print_interval=print_interval,
        device=device,
    )

    data_path = eval_data_path("train", _task_id(dataset.id, AGENT_ID))
    minari_data = minari_col.save(data_path, id=dataset.id, agent_id=AGENT_ID)
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





