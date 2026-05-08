
import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from ice_offline.dataset._spec import BaseDataset
from ice_offline.pipeline import MinariLoader
from ice_offline.agent._spec import EnvSpec
from ice_offline.agent._spec import model_ref
from ice_offline.tools.paths import eval_root



BatchType = dict[str, Any]
TransitionBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OfflineEvalFn = Callable[["RunnerAgent", TransitionBatch], dict[str, float]]
OnlineEvalFn = Callable[[TransitionBatch], dict[str, float]]

class RunnerAgent(Protocol):
    device: str
    def configure(self, env_spec: EnvSpec) -> None: ...
    def act_best(self, observation: Any) -> Any: ...
    def update(self, batch: BatchType) -> None: ...
    def save(self, model_name: str | Path) -> Path: ...
    def load(self, model_name: str | Path) -> None: ...


@dataclass
class TorchBatchOfflineRunner:
    runner_id: str
    batch_size: int
    steps: int
    steps_begin: int = 0
    save_interval: int = 0
    eval_interval: int = 0
    eval_batches: int = 8
    eval_episodes: int = 5
    eval_dir: str = str(eval_root())
    _csv_path: Path = field(init=False, repr=False)
    _metric_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._csv_path = Path(self.eval_dir) / f"{self.runner_id}.csv"
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset: BaseDataset,
        agent: RunnerAgent,
        eval_offline_fns: list[OfflineEvalFn],
        eval_online_fns: list[OnlineEvalFn],
    ) -> None:
        
        batch_loader = MinariLoader(dataset)
        env_spec = EnvSpec(
            observation_shape=batch_loader.observation_shape,
            observation_cardinality=batch_loader.observation_cardinality,
            action_shape=batch_loader.action_shape,
            action_cardinality=batch_loader.action_cardinality,
        )
        agent.configure(env_spec)
        if self.steps_begin > 0:
            agent.load(model_ref(self.runner_id, self.steps_begin))

        for step in range(self.steps_begin + 1,  self.steps + 1):
            batch: BatchType = batch_loader.sample_batch(self.batch_size)
            agent.update(batch)
            
            if self.save_interval > 0 and step % self.save_interval == 0:
                agent.save(model_ref(self.runner_id, step))

            if self.eval_interval > 0 and step % self.eval_interval == 0:
                row: dict[str, float] = {"step": float(step)}
                reduced = self._evaluate_offline(agent, batch_loader, eval_offline_fns)
                row.update(reduced)
                online = self._evaluate_online(agent, eval_online_fns, dataset)
                row.update(online)
                self._append_eval_row(row)

                metrics = [f"step={step}/{self.steps}"]
                for key, value in row.items():
                    if key == "step":
                        continue
                    if key.endswith("_q25") or key.endswith("_q75"):
                        continue
                    if key.endswith("_q50"):
                        metrics.append(f"{key[:-4]}={value:.6f}")
                        continue
                    metrics.append(f"{key}={value:.6f}")
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
        agent: RunnerAgent,
        batch_loader: MinariLoader,
        eval_offline_fns: list[OfflineEvalFn],
    ) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(self.eval_batches):
                batch = batch_loader.sample_batch(self.batch_size)
                o = torch.as_tensor(batch["obs"], device=agent.device)
                a = torch.as_tensor(batch["act"], device=agent.device)
                r = torch.as_tensor(batch["rew"], device=agent.device).view(-1, 1)
                on = torch.as_tensor(batch["next_obs"], device=agent.device)
                d = torch.as_tensor(batch["done"], device=agent.device).view(-1, 1)
                
                transitions: TransitionBatch = (o, a, r, on, d)
                for fn in eval_offline_fns:
                    values = fn(agent, transitions)
                    for k, v in values.items():
                        bucket.setdefault(k, []).append(float(v))
        return self._reduce_bucket_quantiles(bucket)

    def _evaluate_online(
        self,
        agent: RunnerAgent,
        eval_online_fns: list[OnlineEvalFn],
        dataset: BaseDataset,
    ) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        env = dataset.make_eval_env()
        try:
            for _ in range(self.eval_episodes):
                obs, _ = env.reset()
                o = dataset.obs_encode(obs)

                done = False
                obs_list: list[torch.Tensor] = []
                act_list: list[torch.Tensor] = []
                rew_list: list[torch.Tensor] = []
                next_obs_list: list[torch.Tensor] = []
                done_list: list[torch.Tensor] = []
                while not done:
                    act = agent.act_best(o)
                    a = dataset.act_encode(act)

                    next_obs, r, terminated, truncated, _ = env.step(a)
                    on = dataset.obs_encode(next_obs)
                    
                    done = bool(terminated or truncated)

                    obs_list.append(torch.as_tensor(o, device=agent.device))
                    act_list.append(torch.as_tensor(a, device=agent.device))
                    rew_list.append(torch.as_tensor(r, device=agent.device))
                    next_obs_list.append(torch.as_tensor(on, device=agent.device))
                    done_list.append(torch.as_tensor(done, device=agent.device))
                    
                    o = on

                o_batch = torch.stack(obs_list, dim=0)
                a_batch = torch.stack(act_list, dim=0)
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
        return self._reduce_bucket_quantiles(bucket)

    def _reduce_bucket_quantiles(self, bucket: dict[str, list[float]]) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for key, values in bucket.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            q25, q50, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
            reduced[f"{key}_q25"] = float(q25)
            reduced[f"{key}_q50"] = float(q50)
            reduced[f"{key}_q75"] = float(q75)
        return reduced


