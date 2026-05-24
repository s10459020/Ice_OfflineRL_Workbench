import gymnasium as gym
import minari
import numpy as np
import torch

from ice_offline.agent.bc_continuous_deterministic import BCAgentContinuousDeterministic
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.pipeline.minari_collector import MinariCollectorWrapper
from ice_offline.pipeline.state.hopper import HopperStateIO
from ice_offline.pipeline.state_operator.state_collector import StateCollectWrapper
from ice_offline.pipeline.state_operator.state_dataset import StateDataset
from ice_offline.runner.evaluator2 import Evaluator2
from ice_offline.tools.printer import print_stage


TASK_ID = "hopper_simple_bc_deterministic"
DATASET_KEY = "hopper_simple"
OUTPUT_DATASET_ID = "mujoco/hopper/simple-bc-eval-collect-v0"

BATCH_SIZE = 256
STEPS = 200_000
EVAL_INTERVAL = 2_000
EVAL_OFFLINE_N = 8
EVAL_ONLINE_N = 3


def eval_loss_pi(agent: BCAgentContinuousDeterministic, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, _, _, _ = episode_batch
    with torch.no_grad():
        return {"loss_pi": float(agent.loss_actor(o, a).item())}


def eval_reward(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"reward_sum": float(r.sum().item())}


def train(
    task_id: str,
    dataset,
    *,
    eval_env: gym.Env,
    seed: int = 42,
    batch_size: int = 256,
    steps: int = 200_000,
    log_interval: int = 0,
    eval_interval: int = 2_000,
    eval_offline_n: int = 8,
    eval_online_n: int = 3,
    recode_eval: bool = True,
    recode_reset: bool = True,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_loader = MinariLoader(dataset=dataset, seed=seed)

    print_stage("Train BC Deterministic")
    agent = BCAgentContinuousDeterministic(
        obs_size=batch_loader.obs_dim,
        act_size=batch_loader.act_dim,
    )

    evaluator = Evaluator2(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss_pi],
        eval_online_fns=[eval_reward],
        recode_eval=recode_eval,
        recode_reset=recode_reset,
    )

    for step in range(1, steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)
        if log_interval > 0 and step % log_interval == 0:
            print(f"[hopper_simple_bc][step] {step}")
        evaluator.eval_offline(step=step, agent=agent, batch_loader=batch_loader, batch_size=batch_size)
        evaluator.eval_online(step=step, agent=agent, env=eval_env)
        evaluator.print(step)
        evaluator.recode(step)


def collect(
    dataset_key: str,
    task_id: str,
    output_dataset_id: str,
) -> tuple[minari.MinariDataset, StateDataset]:
    dataset = get_dataset(dataset_key)

    base_env = gym.make(dataset.env_id)
    state_col = StateCollectWrapper(base_env, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    train(
        task_id=task_id,
        dataset=dataset,
        eval_env=minari_col,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        eval_interval=EVAL_INTERVAL,
        eval_offline_n=EVAL_OFFLINE_N,
        eval_online_n=EVAL_ONLINE_N,
    )

    minari_data = minari_col.save(output_dataset_id)
    state_data = state_col.save(output_dataset_id)
    minari_col.close()

    return minari_data, state_data


def main() -> None:
    minari_data, state_data = collect(
        dataset_key=DATASET_KEY,
        task_id=TASK_ID,
        output_dataset_id=OUTPUT_DATASET_ID,
    )
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")


if __name__ == "__main__":
    main()
