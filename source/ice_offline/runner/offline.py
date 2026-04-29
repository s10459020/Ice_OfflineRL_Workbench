from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ice_offline.agent._interface import model_ref
from ice_offline.paths import eval_root

BatchType = dict[str, Any]
TransitionBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OfflineEvalFn = Callable[[Any, TransitionBatch], dict[str, float]]
OnlineEvalFn = Callable[[TransitionBatch], dict[str, float]]
EvalEnvFn = Callable[[], Any]


@dataclass
class TorchBatchOfflineRunner:
    obs_encode: Callable[[Any], Any]
    batch_size: int
    train_steps: int
    eval_interval: int
    eval_batches: int = 8
    eval_episodes: int = 5
    model_id: str = "default"
    model_load_step: int = 0
    model_save_interval: int = 0
    eval_dir: str = str(eval_root())
    _csv_path: Path = field(init=False, repr=False)
    _metric_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def train(
        self,
        agent: Any,
        dataset: Any,
        eval_offline_fns: list[OfflineEvalFn] | None = None,
        eval_online_fns: list[OnlineEvalFn] | None = None,
        eval_env_fn: EvalEnvFn | None = None,
    ) -> None:
        dataset_id = str(getattr(dataset, "dataset_id", "minari_dataset")).replace("/", "__")
        self._csv_path = Path(self.eval_dir) / f"{dataset_id}.csv"
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_load_step > 0:
            agent.load(model_ref(self.model_id, self.model_load_step))
        for step in range(1, self.train_steps + 1):
            batch: BatchType = dataset.sample_batch(self.batch_size)
            agent.update(batch, grad_step=step)
            total_step = self.model_load_step + step

            if self.model_save_interval > 0 and total_step % self.model_save_interval == 0:
                agent.save(model_ref(self.model_id, total_step))

            if step % self.eval_interval != 0:
                continue

            row: dict[str, float] = {"step": float(step)}
            if eval_offline_fns:
                reduced = self._evaluate_offline(agent, dataset, eval_offline_fns)
                row.update(reduced)
            if eval_online_fns and eval_env_fn is not None:
                online = self._evaluate_online(agent, eval_online_fns, eval_env_fn)
                row.update(online)
            self._append_eval_row(row)
            metrics = [f"step={step}/{self.train_steps}"] + [
                f"{k}={v:.6f}" for k, v in row.items() if k != "step"
            ]
            print(" ".join(metrics))

    def _append_eval_row(self, row: dict[str, float]) -> None:
        metric_keys = sorted([k for k in row.keys() if k != "step"])
        if not self._metric_keys:
            self._metric_keys = metric_keys
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["step"] + self._metric_keys)
                writer.writeheader()
        out = {"step": int(row["step"])}
        for k in self._metric_keys:
            out[k] = row.get(k, float("nan"))
        with self._csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + self._metric_keys)
            writer.writerow(out)

    def _evaluate_offline(
        self,
        agent: Any,
        dataset: Any,
        eval_offline_fns: list[OfflineEvalFn],
    ) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(self.eval_batches):
                batch = dataset.sample_batch(self.batch_size)
                o = torch.as_tensor(batch["obs"], dtype=torch.float32, device=agent.device)
                a = torch.as_tensor(batch["act"], dtype=torch.long, device=agent.device).view(-1)
                r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=agent.device).view(-1, 1)
                on = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=agent.device)
                d = torch.as_tensor(batch["done"], dtype=torch.float32, device=agent.device).view(-1, 1)
                episode_batch: TransitionBatch = (o, a, r, on, d)
                for fn in eval_offline_fns:
                    values = fn(agent, episode_batch)
                    for k, v in values.items():
                        bucket.setdefault(k, []).append(float(v))
        return {k: float(np.mean(vs)) for k, vs in bucket.items()}

    def _evaluate_online(
        self,
        agent: Any,
        eval_online_fns: list[OnlineEvalFn],
        eval_env_fn: EvalEnvFn,
    ) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        env = eval_env_fn()
        try:
            for _ in range(self.eval_episodes):
                obs, _ = env.reset()
                done = False
                obs_list: list[torch.Tensor] = []
                act_list: list[torch.Tensor] = []
                rew_list: list[torch.Tensor] = []
                next_obs_list: list[torch.Tensor] = []
                done_list: list[torch.Tensor] = []
                while not done:
                    o = self.obs_encode({"image": np.asarray([obs["image"]])})[0]
                    a = int(agent.act(o, epsilon=0.0))
                    next_obs, r, terminated, truncated, _ = env.step(a)
                    done = bool(terminated or truncated)
                    on = self.obs_encode({"image": np.asarray([next_obs["image"]])})[0]
                    obs_list.append(torch.as_tensor(o, dtype=torch.float32, device=agent.device))
                    act_list.append(torch.as_tensor(a, dtype=torch.long, device=agent.device))
                    rew_list.append(torch.as_tensor(float(r), dtype=torch.float32, device=agent.device))
                    next_obs_list.append(torch.as_tensor(on, dtype=torch.float32, device=agent.device))
                    done_list.append(torch.as_tensor(done, dtype=torch.float32, device=agent.device))
                    obs = next_obs
                o_batch = torch.stack(obs_list, dim=0)
                a_batch = torch.stack(act_list, dim=0).view(-1)
                r_batch = torch.stack(rew_list, dim=0).view(-1, 1)
                on_batch = torch.stack(next_obs_list, dim=0)
                d_batch = torch.stack(done_list, dim=0).view(-1, 1)
                episode_batch: TransitionBatch = (o_batch, a_batch, r_batch, on_batch, d_batch)
                for fn in eval_online_fns:
                    values = fn(episode_batch)
                    for k, v in values.items():
                        bucket.setdefault(k, []).append(float(v))
        finally:
            env.close()
        return {k: float(np.mean(vs)) for k, vs in bucket.items()}


