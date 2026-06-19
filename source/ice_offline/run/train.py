import math
import gymnasium as gym
from ice_offline.agent._spec import Agent
from ice_offline.store.eval.collector import EvalCollector

from ice_offline.dataset.base import Dataset
from ice_offline.store.metric.record import MetricRecorder
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.tools.printer import print_stage


def eval(agent: Agent, env: gym.Env, seed: int = 42, count: int = 10) -> list[float]:
    returns: list[float] = []
    for i in range(count):
        agent.set_seed(seed + i)
        o, _ = env.reset(seed=seed + i)
        result = 0.0
        trun = term = False
        while not (trun or term):
            a = agent.act(o)
            o, r, trun, term, _ = env.step(a)
            result += float(r)
        returns.append(result)
    return returns

def train(
    agent: Agent,
    dataset: Dataset,
    *,
    task_id: str | None = None,
    start: int = 0,
    steps: int = 200_000,
    batch_size: int = 256,
    save_interval: int = 20000,
    eval_interval: int = 2000,
    eval_count: int = 10,
    eval_env: gym.Env | None = None,
    print_interval: int = 200,
    seed: int = 42,
) -> None:
    task_id = task_id or _task_id(dataset.id, agent.id)
    eval_env = eval_env or dataset.make_eval_env()

    path = eval_data_path("train", task_id)
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
            returns = eval(agent, eval_col, seed + step, eval_count)
            eval_col.flush(step)
            avg_return = sum(returns) / len(returns)
            print(f"eval step={step} avg_return={avg_return:.6g}")
        
        if step % save_interval == 0 or step == steps:
            agent.save(task_id, step)

    eval_col.save(path)
    eval_col.close()
    return path


if __name__ == "__main__":
    from ice_offline.agent._lookup import make_agent
    from ice_offline.dataset._lookup import make_dataset
    from ice_offline.store.eval.loader import EvalLoader

    device = "cuda:0"
    task_id = "check_run-v0"
    dataset = make_dataset("hopper_medium", device=device)
    agent = make_agent("td3bc", dataset, device=device)
    path = train(agent, dataset, task_id=task_id, steps=50000)

    loader = EvalLoader(path, device=device)
    data = Dataset(path=path, loader=loader, device=device)
    print(f"total_episodes={data.episode_count}")
    print(f"total_steps={data.count}")





