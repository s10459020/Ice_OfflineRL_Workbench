
import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from ice_offline.dataset import BatchLoader
from ice_offline.agent._interface import model_ref
from ice_offline.tools.paths import eval_root
from ice_offline.tools.timing import Timer

BatchType = dict[str, Any]
TransitionBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class RunnerAgent(Protocol):
    device: str
    def act(self, observation: Any, epsilon: float = 0.0) -> Any: ...
    def update(self, batch: BatchType) -> None: ...
    def save(self, model_name: str | Path) -> Path: ...
    def load(self, model_name: str | Path) -> None: ...


OfflineEvalFn = Callable[[RunnerAgent, TransitionBatch], dict[str, float]]
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
    runner_id: str = "default"
    model_load_step: int = 0
    model_save_interval: int = 0
    eval_dir: str = str(eval_root())
    debug_timing: bool = False
    _csv_path: Path = field(init=False, repr=False)
    _metric_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def train(
        self,
        agent: RunnerAgent,
        dataset: BatchLoader,
        eval_offline_fns: list[OfflineEvalFn] | None = None,
        eval_online_fns: list[OnlineEvalFn] | None = None,
        eval_env_fn: EvalEnvFn | None = None,
    ) -> None:
        run_id = str(self.runner_id).replace("/", "__")
        self._csv_path = Path(self.eval_dir) / f"{run_id}.csv"
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        metric_names = (
            "sample",
            "update",
            "save",
            "eval_offline",
            "eval_online",
            "step_non_eval",
            "step_eval",
        )

        def summary_line() -> str:
            parts: list[str] = []
            for metric_name in metric_names:
                avg_ms = Timer.get(f"runner.{metric_name}")
                parts.append(f"{metric_name}={avg_ms:.3f}")
            return "perf_ms(avg100) " + " ".join(parts)

        def timer_call(metric_name: str, callback: Callable[[], Any]) -> Any:
            if self.debug_timing:
                _, value = Timer.record(f"runner.{metric_name}", callback)
                return value
            return callback()

        if self.model_load_step > 0:
            agent.load(model_ref(self.runner_id, self.model_load_step))
        for step in range(1, self.train_steps + 1):
            def run_train_core() -> None:
                batch: BatchType = timer_call("sample", lambda: dataset.sample_batch(self.batch_size))
                timer_call("update", lambda: agent.update(batch))
                model_step = self.model_load_step + step
                if self.model_save_interval > 0 and model_step % self.model_save_interval == 0:
                    timer_call("save", lambda: agent.save(model_ref(self.runner_id, model_step)))

            if step % self.eval_interval != 0:
                timer_call("step_non_eval", run_train_core)
                continue

            def run_eval_step() -> dict[str, float]:
                run_train_core()
                row: dict[str, float] = {"step": float(step)}
                if eval_offline_fns:
                    reduced = timer_call("eval_offline", lambda: self._evaluate_offline(agent, dataset, eval_offline_fns))
                    row.update(reduced)
                if eval_online_fns and eval_env_fn is not None:
                    online = timer_call("eval_online", lambda: self._evaluate_online(agent, eval_online_fns, eval_env_fn))
                    row.update(online)
                self._append_eval_row(row)
                return row

            row = timer_call("step_eval", run_eval_step)
            metrics = [f"step={step}/{self.train_steps}"]
            for key, value in row.items():
                if key == "step":
                    continue
                if key.endswith("_q25") or key.endswith("_q75"):
                    continue
                if key.endswith("_q50"):
                    metrics.append(f"{key[:-4]}={value:.6f}")
                    continue
                metrics.append(f"{key}={value:.6f}")
            if self.debug_timing:
                print(" ".join(metrics) + " " + summary_line())
            else:
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
        dataset: BatchLoader,
        eval_offline_fns: list[OfflineEvalFn],
    ) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(self.eval_batches):
                batch = dataset.sample_batch(self.batch_size)
                o = torch.as_tensor(batch["obs"], dtype=torch.float32, device=agent.device)
                act_dtype = torch.long if np.issubdtype(np.asarray(batch["act"]).dtype, np.integer) else torch.float32
                a = torch.as_tensor(batch["act"], dtype=act_dtype, device=agent.device)
                if act_dtype == torch.long:
                    a = a.view(-1)
                r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=agent.device).view(-1, 1)
                on = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=agent.device)
                d = torch.as_tensor(batch["done"], dtype=torch.float32, device=agent.device).view(-1, 1)
                episode_batch: TransitionBatch = (o, a, r, on, d)
                for fn in eval_offline_fns:
                    values = fn(agent, episode_batch)
                    for k, v in values.items():
                        bucket.setdefault(k, []).append(float(v))
        return self._reduce_bucket_quantiles(bucket)

    def _evaluate_online(
        self,
        agent: RunnerAgent,
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
                    obs_batch = (
                        {k: np.asarray([v]) for k, v in obs.items()}
                        if isinstance(obs, dict)
                        else np.asarray([obs])
                    )
                    o = self.obs_encode(obs_batch)[0]
                    try:
                        a_raw = agent.act(o, epsilon=0.0)
                    except TypeError:
                        try:
                            a_raw = agent.act(o, greedy=True)
                        except TypeError:
                            a_raw = agent.act(o)
                    if hasattr(env.action_space, "n"):
                        env_action = int(np.asarray(a_raw).item())
                    else:
                        env_action = np.asarray(a_raw, dtype=np.float32).reshape(env.action_space.shape)
                    next_obs, r, terminated, truncated, _ = env.step(env_action)
                    done = bool(terminated or truncated)
                    next_obs_batch = (
                        {k: np.asarray([v]) for k, v in next_obs.items()}
                        if isinstance(next_obs, dict)
                        else np.asarray([next_obs])
                    )
                    on = self.obs_encode(next_obs_batch)[0]
                    act_dtype = torch.long if np.issubdtype(np.asarray(a_raw).dtype, np.integer) else torch.float32
                    obs_list.append(torch.as_tensor(o, dtype=torch.float32, device=agent.device))
                    act_list.append(torch.as_tensor(a_raw, dtype=act_dtype, device=agent.device))
                    rew_list.append(torch.as_tensor(float(r), dtype=torch.float32, device=agent.device))
                    next_obs_list.append(torch.as_tensor(on, dtype=torch.float32, device=agent.device))
                    done_list.append(torch.as_tensor(done, dtype=torch.float32, device=agent.device))
                    obs = next_obs
                o_batch = torch.stack(obs_list, dim=0)
                a_batch = torch.stack(act_list, dim=0)
                if a_batch.ndim == 1:
                    a_batch = a_batch.view(-1)
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


