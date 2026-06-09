import math
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
from ice_offline.store.eval.record import Evaluator
from ice_offline.store.metric.record import MetricRecorder
from ice_offline.config.paths import data_path_train
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
AGENT_ID = "scas_mean"


def print_latest(step: int, recorder: MetricRecorder) -> None:
    metrics = recorder.last
    parts = [f"{name}={value:.6g}" for name, value in metrics.items()]
    print(f"train step={step}", *parts)


def update_dynamic_with_record(recorder: MetricRecorder, dynamics: ScasDynamic, batch: Batch) -> None:
    loss_dynamic = dynamics.loss_dynamic(batch)
    recorder.add("loss_dynamic", loss_dynamic)
    recorder.add_grad_norm("grad_dynamic", loss_dynamic, dynamics.model.parameters())
    recorder.flush()
    dynamics.update(batch)


def update_agent_with_record(recorder: MetricRecorder, agent: ScasMeanAgent, batch: Batch) -> None:
    actor_update = (agent.update_step + 1) % agent.update_actor_interval == 0
    loss_critic = agent.loss_critic(batch)

    recorder.add("loss_critic", loss_critic)
    recorder.add_grad_norm("grad_critic", loss_critic, agent.critic.parameters())

    if actor_update:
        loss_td3 = agent.loss_td3(batch)
        loss_correction = agent.loss_correction(batch)
        loss_actor = (1.0 - agent.lmbda) * loss_td3 + agent.lmbda * loss_correction
        recorder.add("loss_td3", loss_td3)
        recorder.add_grad_norm("grad_td3", loss_td3, agent.actor.parameters())
        recorder.add("loss_correction", loss_correction)
        recorder.add_grad_norm("grad_correction", loss_correction, agent.actor.parameters())
        recorder.add("loss_actor", loss_actor)
        recorder.add_grad_norm("grad_actor", loss_actor, agent.actor.parameters())

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


    print_stage("Train SCAS Mean Dynamics")
    dynamics = ScasDynamic(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
    )
    dynamics.agent_name = f"{AGENT_ID}_dynamics"
    model_recorder = MetricRecorder(dataset.id, dynamics.agent_name)
    for step in range(1, model_steps + 1):
        batch = dataset.sample_batch(batch_size)
        update_dynamic_with_record(model_recorder, dynamics, batch)
        if print_interval > 0 and step % print_interval == 0:
            print_latest(step, model_recorder)
        if step % save_interval == 0 or step == model_steps:
            dynamics.save(dataset.id, step)


    print_stage("Train SCAS Mean Agent")
    agent = ScasMeanAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        dynamics=dynamics,
        device=device,
    )
    agent.set_seed(seed)

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
    model_steps: int = MODEL_STEPS,
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
        model_steps=model_steps,
        steps=steps,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        save_interval=save_interval,
        print_interval=print_interval,
        device=device,
    )

    data_path = data_path_train(dataset.id, AGENT_ID)
    minari_data = minari_col.save(data_path, id=dataset.id, agent_id=AGENT_ID)
    state_data = state_col.save(data_path)
    minari_col.close()

    return minari_data, state_data


if __name__ == "__main__":
    dataset = HopperSimpleDataset(device=DEVICE)
    minari_data, state_data = collect(dataset=dataset)
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")









