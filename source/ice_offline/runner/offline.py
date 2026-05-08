import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

from ice_offline.agent._spec import EnvSpec
from ice_offline.agent._spec import model_ref
from ice_offline.dataset._spec import BaseDataset
from ice_offline.pipeline import MinariLoader
from ice_offline.tools.paths import eval_root


BatchType = dict[str, Any]
TransitionBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OfflineEvalFn = Callable[["RunnerAgent", TransitionBatch], dict[str, float]]
OnlineEvalFn = Callable[[TransitionBatch], dict[str, float]]


class EarlyStopEvent(Protocol):
    def should_stop(self, metrics: dict[str, list[float]]) -> bool: ...


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
    _eval_dir_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._eval_dir_path = Path(self.eval_dir) / self.runner_id
        self._eval_dir_path.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset: BaseDataset,
        agent: RunnerAgent,
        eval_offline_fns: list[OfflineEvalFn],
        eval_online_fns: list[OnlineEvalFn],
        early_stop_events: list[EarlyStopEvent] | None = None,
    ) -> None:
        minari_loader = MinariLoader(dataset)
        env_spec = EnvSpec(
            observation_shape=minari_loader.observation_shape,
            observation_cardinality=minari_loader.observation_cardinality,
            action_shape=minari_loader.action_shape,
            action_cardinality=minari_loader.action_cardinality,
        )
        agent.configure(env_spec)
        if self.steps_begin > 0:
            agent.load(model_ref(self.runner_id, self.steps_begin))

        for step in range(self.steps_begin + 1, self.steps + 1):
            batch: BatchType = minari_loader.sample_batch(self.batch_size)
            agent.update(batch)

            if self.save_interval > 0 and step % self.save_interval == 0:
                agent.save(model_ref(self.runner_id, step))

            if self.eval_interval > 0 and step % self.eval_interval == 0:
                offline_metrics = self._evaluate_offline(agent, minari_loader, eval_offline_fns)
                online_metrics = self._evaluate_online(agent, eval_online_fns, dataset)
                metrics = {**offline_metrics, **online_metrics}
                self._append_eval_step(step, metrics)

                summary = [f"step={step}/{self.steps}"]
                for key in sorted(metrics.keys()):
                    summary.append(f"{key}={sum(metrics[key]) / len(metrics[key]):.6g}")
                print(" ".join(summary))

                if early_stop_events is not None:
                    should_stop = False
                    for early_stop_event in early_stop_events:
                        if early_stop_event.should_stop(metrics):
                            should_stop = True
                            break
                    if should_stop:
                        print(f"early_stop=True step={step}")
                        break

    def _append_eval_step(self, step: int, metrics: dict[str, list[float]]) -> None:
        for metric_key, values in metrics.items():
            self._append_metric_row(step, metric_key, values)

    def _append_metric_row(self, step: int, metric_key: str, values: list[float]) -> None:
        csv_path = self._eval_dir_path / f"{metric_key}.csv"
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["step"] + [str(i) for i in range(1, len(values) + 1)])
            writer.writerow([int(step)] + values)

    def _evaluate_offline(
        self,
        agent: RunnerAgent,
        minari_loader: MinariLoader,
        eval_offline_fns: list[OfflineEvalFn],
    ) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(self.eval_batches):
                batch = minari_loader.sample_batch(self.batch_size)
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

    def _evaluate_online(
        self,
        agent: RunnerAgent,
        eval_online_fns: list[OnlineEvalFn],
        dataset: BaseDataset,
    ) -> dict[str, list[float]]:
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

                obs_batch = torch.stack(obs_list, dim=0)
                act_batch = torch.stack(act_list, dim=0)
                rew_batch = torch.stack(rew_list, dim=0).view(-1, 1)
                next_obs_batch = torch.stack(next_obs_list, dim=0)
                done_batch = torch.stack(done_list, dim=0).view(-1, 1)

                episode_batch: TransitionBatch = (
                    obs_batch,
                    act_batch,
                    rew_batch,
                    next_obs_batch,
                    done_batch,
                )
                for eval_fn in eval_online_fns:
                    values = eval_fn(episode_batch)
                    for key, value in values.items():
                        bucket.setdefault(key, []).append(float(value))
        finally:
            env.close()
        return bucket
