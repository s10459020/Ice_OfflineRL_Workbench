import gymnasium as gym
import minari
import numpy as np
import torch
import math

from ice_offline.agent.td3bc import TD3BCAgent
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
SAVE_INTERVAL = math.ceil(STEPS/10)
EVAL_INTERVAL = math.ceil(STEPS/100)
PRINT_INTERVAL = math.ceil(STEPS/1000)

SEED = 42
BATCH_SIZE = 256
EVAL_EPISODES = 10
DEVICE = "cuda:0"
AGENT_ID = "td3bc"


def update_with_record(recorder: MetricRecorder, agent: TD3BCAgent, batch: Batch) -> None:
    agent.update_step += 1

    loss_critic = agent.loss_critic(batch)
    recorder.add("loss_critic", loss_critic)
    recorder.add_grad_norm("grad_critic", loss_critic, agent.critic.parameters())

    agent.critic_optimizer.zero_grad()
    loss_critic.backward()
    agent.critic_optimizer.step()

    if agent.update_step % agent.update_actor_interval == 0:
        loss_td3 = agent.loss_td3(batch)
        loss_bc = agent.loss_bc(batch)
        loss_actor = agent.alpha * loss_td3 + loss_bc

        recorder.add("loss_td3", loss_td3)
        recorder.add_grad_norm("grad_td3", loss_td3, agent.actor.parameters())
        recorder.add("loss_bc", loss_bc)
        recorder.add_grad_norm("grad_bc", loss_bc, agent.actor.parameters())
        recorder.add("loss_actor", loss_actor)
        recorder.add_grad_norm("grad_actor", loss_actor, agent.actor.parameters())

        agent.actor_optimizer.zero_grad()
        loss_actor.backward()
        agent.actor_optimizer.step()
        agent.critic.update_target_soft()
        agent.actor.update_target_soft()

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

    print_stage("Train TD3BC")
    agent = TD3BCAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=device,
    )

    recorder = MetricRecorder(dataset.id, AGENT_ID)
    evaluator = Evaluator(dataset.id, AGENT_ID, episodes=eval_episodes)

    for step in range(1, steps + 1):
        batch = dataset.sample_batch(batch_size)
        update_with_record(recorder, agent, batch)
        if print_interval > 0 and step % print_interval == 0:
            metrics = recorder.last
            parts = [f"{name}={value:.6g}" for name, value in metrics.items()]
            print(f"train step={step}", *parts)
        if eval_interval > 0 and step % eval_interval == 0:
            avg_return = evaluator.eval(step, agent, eval_env)
            print(f"eval step={step} avg_return={avg_return:.6g}")
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






