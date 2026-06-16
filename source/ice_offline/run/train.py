import math
import gymnasium as gym
from ice_offline.agent._spec import Agent
from ice_offline.run.test import eval
from ice_offline.store.eval.collector import EvalCollector

from ice_offline.dataset.base import Dataset
from ice_offline.store.metric.record import MetricRecorder
from ice_offline.config.paths import _task_id
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

def train(
    agent: Agent,
    dataset: Dataset,
    *,
    task_id: str | None = None,
    start: int = 0,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    eval_interval: int = EVAL_INTERVAL,
    eval_episodes: int = EVAL_EPISODES,
    eval_env: gym.Env | None = None,
    save_interval: int = SAVE_INTERVAL,
    print_interval: int = PRINT_INTERVAL,
    seed: int = SEED,
) -> None:
    task_id = task_id or _task_id(dataset.id, agent.id)
    eval_env = eval_env or dataset.make_eval_env()

    path = data_path_train(task_id)
    resume_path = path if start > 0 else None
    eval_col = EvalCollector(eval_env, resume_path=resume_path)
    recorder = MetricRecorder(task_id, initialized=start > 0)

    print_stage(f"Train {agent.id} in {dataset.id}")
    for step in range(start + 1, steps + 1):
        # seed
        now_seed = seed + step
        agent.set_seed(now_seed)
        dataset.set_seed(now_seed)

        # run
        batch = dataset.sample_batch(batch_size)
        metrics = agent.update_with_metrics(batch)

        # record
        for name, value in metrics.items():
            recorder.add(name, value)
        recorder.flush(step)
        
        # functional
        if print_interval > 0 and step % print_interval == 0:
            metrics = recorder.last
            parts = [f"{name}={value:.6g}" for name, value in metrics.items()]
            print(f"train step={step}", *parts)
        
        if eval_interval > 0 and step % eval_interval == 0:
            eval_seed = seed + step
            returns = eval(agent, eval_col, eval_episodes, eval_seed)
            eval_col.flush(step)
            avg_return = sum(returns) / len(returns)
            print(f"eval step={step} avg_return={avg_return:.6g}")
        
        if step % save_interval == 0 or step == steps:
            agent.save(task_id, step)

    eval_col.save(path)
    eval_col.close()
    return path


if __name__ == "__main__":
    from ice_offline.dataset.hopper_simple import HopperSimpleDataset
    from ice_offline.agent.bc_deterministic import BCDeterministicAgent
    from ice_offline.store.eval.loader import EvalLoader

    dataset = HopperSimpleDataset(device=DEVICE)
    agent = BCDeterministicAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=DEVICE,
    )

    path = train(agent, dataset)

    loader = EvalLoader(path, device=DEVICE)
    data = Dataset(path=path, loader=loader, device=DEVICE)
    print(f"total_episodes={data.episode_count}")
    print(f"total_steps={data.count}")





