from pathlib import Path
from typing import Callable

import torch

from ice_offline.tools.paths import eval_root


TransitionBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OfflineEvalFn = Callable[[object, TransitionBatch], dict[str, float]]
OnlineEvalFn = Callable[[TransitionBatch], dict[str, float]]


class Evaluator2:
    def __init__(
        self,
        runner_id: str,
        eval_interval: int,
        eval_offline_n: int = 0,
        eval_online_n: int = 0,
        eval_offline_fns: list[OfflineEvalFn] = [],
        eval_online_fns: list[OnlineEvalFn] = [],
        recode_eval: bool = True,
        recode_reset: bool = False,
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
            self.eval_dir = Path(eval_root()) / runner_id
            self.eval_dir.mkdir(parents=True, exist_ok=True)
            if recode_reset:
                for p in self.eval_dir.glob("*"):
                    if p.is_file():
                        p.unlink()

    def should_eval(self, step: int) -> bool:
        return self.eval_interval > 0 and step % self.eval_interval == 0

    def evaluate_offline(
        self,
        agent,
        batch_loader,
        batch_size: int,
    ) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        with torch.no_grad():
            for _ in range(self.eval_offline_n):
                batch = batch_loader.sample_batch(batch_size)
                obs = torch.as_tensor(batch["obs"], device=agent.device)
                act = torch.as_tensor(batch["act"], device=agent.device)
                rew = torch.as_tensor(batch["rew"], device=agent.device).view(-1, 1)
                next_obs = torch.as_tensor(batch["next_obs"], device=agent.device)
                done = torch.as_tensor(batch["done"], device=agent.device).view(-1, 1)
                transitions: TransitionBatch = (obs, act, rew, next_obs, done)
                for eval_fn in self.eval_offline_fns:
                    values = eval_fn(agent, transitions)
                    for key, value in values.items():
                        bucket.setdefault(key, []).append(float(value))
        return bucket

    def _collect_online(
        self,
        agent,
        env,
    ) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        for _ in range(self.eval_online_n):
            obs, _ = env.reset()
            done = False

            obs_list: list[torch.Tensor] = []
            act_list: list[torch.Tensor] = []
            rew_list: list[torch.Tensor] = []
            next_obs_list: list[torch.Tensor] = []
            done_list: list[torch.Tensor] = []

            while not done:
                act = agent.act_best(obs)
                next_obs, reward, terminated, truncated, _ = env.step(act)
                done = bool(terminated or truncated)

                obs_list.append(torch.as_tensor(obs, device=agent.device))
                act_list.append(torch.as_tensor(act, device=agent.device))
                rew_list.append(torch.as_tensor(reward, device=agent.device))
                next_obs_list.append(torch.as_tensor(next_obs, device=agent.device))
                done_list.append(torch.as_tensor(done, device=agent.device))
                obs = next_obs

            episode_batch: TransitionBatch = (
                torch.stack(obs_list, dim=0),
                torch.stack(act_list, dim=0),
                torch.stack(rew_list, dim=0).view(-1, 1),
                torch.stack(next_obs_list, dim=0),
                torch.stack(done_list, dim=0).view(-1, 1),
            )
            for eval_fn in self.eval_online_fns:
                values = eval_fn(episode_batch)
                for key, value in values.items():
                    bucket.setdefault(key, []).append(float(value))
        return bucket

    def eval_offline(
        self,
        step: int,
        agent,
        batch_loader=None,
        batch_size: int = 0,
    ) -> bool:
        if not self.should_eval(step):
            return False
        if self.last_eval_step != step:
            self.last_evals = {}
            self.last_eval_step = step
        evals: dict[str, list[float]] = {}
        if self.eval_offline_fns:
            offline_evals = self.evaluate_offline(agent, batch_loader, batch_size)
            evals.update(offline_evals)
        self.last_evals.update(evals)
        return True

    def eval_online(
        self,
        step: int,
        agent,
        env,
    ) -> bool:
        if not self.should_eval(step):
            return False
        if self.last_eval_step != step:
            self.last_evals = {}
            self.last_eval_step = step
        evals: dict[str, list[float]] = {}
        if self.eval_online_fns:
            online_evals = self._collect_online(agent, env)
            evals.update(online_evals)
        self.last_evals.update(evals)
        return True

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
