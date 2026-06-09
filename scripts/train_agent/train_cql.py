import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.cql import CQLAgent
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


BATCH_SIZE = 256
STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_EPISODES = 3
SAVE_INTERVAL = 20_000
PRINT_INTERVAL = 10
SEED = 42
DEVICE = "cuda:0"
AGENT_ID = "cql"


def print_latest(step: int, recorder: MetricRecorder) -> None:
    metrics = recorder.history[-1]
    parts = [f"{name}={value:.6g}" for name, value in metrics.items()]
    print(f"train step={step}", *parts)


def update_with_record(recorder: MetricRecorder, agent: CQLSoftQAgent, batch: Batch) -> None:
    o, _, _, _, _ = batch
    loss_td = agent.loss_td(batch)
    loss_suppress = agent.loss_suppress(batch)
    loss_multiplier = agent.multiplier.loss(loss_suppress.detach())

    recorder.add("loss_td", loss_td)
    recorder.add_grad_norm("grad_td", loss_td, agent.critic.parameters())
    recorder.add("loss_suppress", loss_suppress.sum())
    recorder.add_grad_norm("grad_suppress", loss_suppress.sum(), agent.critic.parameters())
    recorder.add("loss_multiplier", loss_multiplier)
    recorder.add_grad_norm("grad_multiplier", loss_multiplier, agent.multiplier.parameters())

    agent.multiplier.optimizer.zero_grad()
    loss_multiplier.backward()
    agent.multiplier.optimizer.step()

    loss_critic = agent.loss_critic_with_suppress(batch, loss_suppress)
    recorder.add("loss_critic", loss_critic)
    recorder.add_grad_norm("grad_critic", loss_critic, agent.critic.parameters())

    agent.critic_optimizer.zero_grad()
    loss_critic.backward()
    agent.critic_optimizer.step()

    a, log_prob = agent.actor.sample(o)
    loss_temperature = agent.temp.loss(log_prob)
    recorder.add("loss_temperature", loss_temperature)
    recorder.add_grad_norm("grad_temperature", loss_temperature, agent.temp.parameters())

    agent.temp.optimizer.zero_grad()
    loss_temperature.backward()
    agent.temp.optimizer.step()

    loss_actor = agent.loss_actor_with_sample(batch, a, log_prob)
    recorder.add("loss_actor", loss_actor)
    recorder.add_grad_norm("grad_actor", loss_actor, agent.actor.parameters())

    agent.actor_optimizer.zero_grad()
    loss_actor.backward()
    agent.actor_optimizer.step()
    agent.critic.update_target_soft()
    recorder.flush()


def train(
    dataset: Dataset,
    *,
    steps: int = STEPS,
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

    print_stage("Train CQL")
    agent = CQLAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
    )

    recorder = MetricRecorder(dataset.id, AGENT_ID)
    evaluator = Evaluator(dataset.id, AGENT_ID, episodes=eval_episodes)

    for step in range(1, steps + 1):
        batch = dataset.sample_batch(batch_size)
        update_with_record(recorder, agent, batch)
        if eval_interval > 0 and step % eval_interval == 0:
            avg_return = evaluator.eval(step, agent, eval_env)
            print(f"eval step={step} avg_return={avg_return:.6g}")  
        if print_interval > 0 and step % print_interval == 0:
            print_latest(step, recorder)
        if step % save_interval == 0 or step == steps:
            agent.save(dataset.id, step)

    evaluator.save()
    recorder.save()


def collect(
    dataset: Dataset,
    *,
    steps: int = STEPS,
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








