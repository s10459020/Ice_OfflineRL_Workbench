import csv
from pathlib import Path
from typing import Callable

import torch


TransitionBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OfflineEvalFn = Callable[[object, TransitionBatch], dict[str, float]]
OnlineEvalFn = Callable[[TransitionBatch], dict[str, float]]


class RunnerEvaluator:
    def __init__(
        self,
        runner_id: str,
        eval_dir: str,
        eval_batches: int,
        eval_episodes: int,
    ) -> None:
        self.eval_batches = eval_batches
        self.eval_episodes = eval_episodes
        self.eval_dir_path = Path(eval_dir) / runner_id
        self.eval_dir_path.mkdir(parents=True, exist_ok=True)

    def evaluate_offline(self, agent, minari_loader, eval_offline_fns: list[OfflineEvalFn], batch_size: int) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(self.eval_batches):
                batch = minari_loader.sample_batch(batch_size)
                obs = torch.as_tensor(batch["obs"], device=agent.device)
                act = torch.as_tensor(batch["act"], device=agent.device)
                rew = torch.as_tensor(batch["rew"], device=agent.device).view(-1, 1)
                next_obs = torch.as_tensor(batch["next_obs"], device=agent.device)
                done = torch.as_tensor(batch["done"], device=agent.device).view(-1, 1)
                transitions: TransitionBatch = (obs, act, rew, next_obs, done)

                for eval_fn in eval_offline_fns:
                    values = eval_fn(agent, transitions)
                    for key, value in values.items():
                        bucket.setdefault(key, []).append(float(value))
        return bucket

    def evaluate_online(self, agent, dataset, eval_online_fns: list[OnlineEvalFn]) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        env = dataset.make_eval_env()
        try:
            for _ in range(self.eval_episodes):
                obs, _ = env.reset()
                encoded_obs = dataset.obs_encode(obs)

                done = False
                obs_list: list[torch.Tensor] = []
                act_list: list[torch.Tensor] = []
                rew_list: list[torch.Tensor] = []
                next_obs_list: list[torch.Tensor] = []
                done_list: list[torch.Tensor] = []

                while not done:
                    act_policy = agent.act_best(encoded_obs)
                    act_env = dataset.act_encode(act_policy)
                    next_obs_raw, reward, terminated, truncated, _ = env.step(act_env)
                    next_encoded_obs = dataset.obs_encode(next_obs_raw)
                    done = bool(terminated or truncated)

                    obs_list.append(torch.as_tensor(encoded_obs, device=agent.device))
                    act_list.append(torch.as_tensor(act_env, device=agent.device))
                    rew_list.append(torch.as_tensor(reward, device=agent.device))
                    next_obs_list.append(torch.as_tensor(next_encoded_obs, device=agent.device))
                    done_list.append(torch.as_tensor(done, device=agent.device))
                    encoded_obs = next_encoded_obs

                episode_batch: TransitionBatch = (
                    torch.stack(obs_list, dim=0),
                    torch.stack(act_list, dim=0),
                    torch.stack(rew_list, dim=0).view(-1, 1),
                    torch.stack(next_obs_list, dim=0),
                    torch.stack(done_list, dim=0).view(-1, 1),
                )
                for eval_fn in eval_online_fns:
                    values = eval_fn(episode_batch)
                    for key, value in values.items():
                        bucket.setdefault(key, []).append(float(value))
        finally:
            env.close()
        return bucket

    def append_eval_step(self, step: int, metrics: dict[str, list[float]]) -> None:
        for metric_key, values in metrics.items():
            csv_path = self.eval_dir_path / f"{metric_key}.csv"
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["step"] + [str(i) for i in range(1, len(values) + 1)])
                writer.writerow([int(step)] + values)

    def summary_text(self, step: int, steps: int, metrics: dict[str, list[float]]) -> str:
        parts = [f"step={step}/{steps}"]
        for key in sorted(metrics.keys()):
            parts.append(f"{key}={sum(metrics[key]) / len(metrics[key]):.6g}")
        return " ".join(parts)
