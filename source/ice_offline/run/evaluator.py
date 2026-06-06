from pathlib import Path
from typing import Callable

import torch

from ice_offline.dataset._types import Batch
from ice_offline.config.paths import EVALS_ROOT


OfflineEvalFn = Callable[[object, Batch], dict[str, float]]
OnlineEvalFn = Callable[[Batch], dict[str, float]]


class Evaluator:
    def __init__(
        self,
        runner_id: str,
        eval_interval: int = 1,
        eval_offline_n: int = 0,
        eval_online_n: int = 0,
        eval_offline_fns: list[OfflineEvalFn] = [],
        eval_online_fns: list[OnlineEvalFn] = [],
        recode_eval: bool = True,
        recode_reset: bool = True,
    ) -> None:
        self.eval_interval = eval_interval
        self.eval_offline_n = eval_offline_n
        self.eval_online_n = eval_online_n
        self.eval_offline_fns = eval_offline_fns
        self.eval_online_fns = eval_online_fns
        self.recode_eval = recode_eval
        self.last_evals: dict[str, list[float]] = {}
        self.last_eval_step: int = -1
        self.eval_dir = None
        if self.recode_eval:
            self.eval_dir = Path(EVALS_ROOT) / runner_id
            self.eval_dir.mkdir(parents=True, exist_ok=True)
            if recode_reset:
                for p in self.eval_dir.glob("*"):
                    if p.is_file():
                        p.unlink()

    def should_eval(self, step: int) -> bool:
        return self.eval_interval > 0 and step % self.eval_interval == 0

    def _eval_offline(
        self,
        agent,
        eval_offline_fn: OfflineEvalFn,
        batch_loader,
        batch_size: int,
    ) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(self.eval_offline_n):
                batch = batch_loader.sample_batch(batch_size)
                values = eval_offline_fn(agent, batch)
                for key, value in values.items():
                    bucket.setdefault(key, []).append(float(value))
        return bucket

    def _eval_online(
        self,
        agent,
        eval_online_fn: OnlineEvalFn,
        eval_env,
    ) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        for _ in range(self.eval_online_n):
            obs, _ = eval_env.reset()
            done = False

            obs_list: list[torch.Tensor] = []
            act_list: list[torch.Tensor] = []
            rew_list: list[torch.Tensor] = []
            next_obs_list: list[torch.Tensor] = []
            done_list: list[torch.Tensor] = []

            while not done:
                act = agent.act_best(obs)
                next_obs, reward, terminated, truncated, _ = eval_env.step(act)
                done = bool(terminated or truncated)

                obs_list.append(torch.as_tensor(obs, device=agent.device))
                act_list.append(torch.as_tensor(act, device=agent.device))
                rew_list.append(torch.as_tensor(reward, device=agent.device))
                next_obs_list.append(torch.as_tensor(next_obs, device=agent.device))
                done_list.append(torch.as_tensor(done, device=agent.device))
                obs = next_obs

            episode_batch = (
                torch.stack(obs_list, dim=0).float(),
                torch.stack(act_list, dim=0).float(),
                torch.stack(rew_list, dim=0).float().view(-1, 1),
                torch.stack(next_obs_list, dim=0).float(),
                torch.stack(done_list, dim=0).float().view(-1, 1),
            )
            values = eval_online_fn(episode_batch)
            for key, value in values.items():
                bucket.setdefault(key, []).append(float(value))
        return bucket

    def eval(
        self,
        step: int,
        agent,
        batch_loader=None,
        batch_size: int = 0,
        eval_env=None,
    ) -> None:
        if not self.should_eval(step):
            return False
        self.last_evals = {}
        self.last_eval_step = step

        if self.eval_offline_fns:
            for eval_fn in self.eval_offline_fns:
                evals = self._eval_offline(
                    agent=agent,
                    eval_offline_fn=eval_fn,
                    batch_loader=batch_loader,
                    batch_size=batch_size,
                )
                for key, values in evals.items():
                    self.last_evals.setdefault(key, []).extend(values)
        if self.eval_online_fns:
            for eval_fn in self.eval_online_fns:
                evals = self._eval_online(
                    agent=agent,
                    eval_online_fn=eval_fn,
                    eval_env=eval_env,
                )
                for key, values in evals.items():
                    self.last_evals.setdefault(key, []).extend(values)

    def recode(self, step: int) -> None:
        if not self.should_eval(step):
            return
        if not self.recode_eval:
            return
        for eval_id, values in self.last_evals.items():
            path = self.eval_dir / f"{eval_id}.csv"
            is_new = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as f:
                if is_new:
                    header = "step," + ",".join(str(i) for i in range(1, len(values) + 1))
                    f.write(f"{header}\n")
                row = ",".join(str(float(v)) for v in values)
                f.write(f"{step},{row}\n")

    def print(self, step: int) -> None:
        if not self.should_eval(step):
            return
        parts: list[str] = []
        for key in sorted(self.last_evals.keys()):
            values = self.last_evals[key]
            parts.append(f"{key}={sum(values) / len(values):.6g}")
        print(f"eval step={step}", *parts)


