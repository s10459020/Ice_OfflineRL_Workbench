import csv
from pathlib import Path

import numpy as np
import torch

from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.tools.paths import eval_root
from ice_offline.tools.printer import print_stage


def _as_tensors(batch: dict[str, np.ndarray], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    a = torch.as_tensor(batch["act"], dtype=torch.float32, device=device)
    r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=device).view(-1, 1)
    sn = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
    d = torch.as_tensor(batch["done"], dtype=torch.float32, device=device).view(-1, 1)
    return s, a, r, sn, d


def _append_csv(path: Path, row: list[float | int], header: list[str]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def train(
    batch_loader: MinariLoader,
    *,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 256,
    dynamics_steps: int = 200_000,
    agent_steps: int = 500_000,
    tau: float = 0.005,
    max_action: float = 1.0,
    gamma: float = 0.99,
    policy_freq: int = 2,
    beta: float = 3e-3,
    alpha: float = 5.0,
    lmbda: float = 0.25,
    log_interval: int = 0,
    eval_interval: int = 0,
    eval_batches: int = 4,
    eval_tag: str = "scas",
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    dynamics_csv = Path(eval_root()) / f"{eval_tag}_dynamics_eval.csv"
    agent_csv = Path(eval_root()) / f"{eval_tag}_agent_eval.csv"
    dynamics_csv.parent.mkdir(parents=True, exist_ok=True)

    print_stage("Train Dynamics")
    dynamics = ScasDynamic(
        obs_dim=batch_loader.obs_dim,
        act_dim=batch_loader.act_dim,
        learning_rate=1e-3,
        device=device,
    )
    for step in range(1, dynamics_steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        dynamics.update(batch)
        if log_interval > 0 and step % log_interval == 0:
            print(f"[dynamics][train] step={step}/{dynamics_steps}")
        if eval_interval > 0 and step % eval_interval == 0:
            loss_sum = 0.0
            for _ in range(eval_batches):
                e_batch = batch_loader.sample_batch(batch_size)
                s, a, _, sn, _ = _as_tensors(e_batch, device)
                with torch.no_grad():
                    loss_sum += float(dynamics.loss_dynamic(s, a, sn).item())
            loss_dynamic = loss_sum / eval_batches
            print(f"[dynamics][eval] step={step}/{dynamics_steps} loss_dynamic={loss_dynamic:.6f}")
            _append_csv(
                dynamics_csv,
                [step, loss_dynamic],
                ["step", "loss_dynamic"],
            )

    print_stage("Train Agent")
    agent = ScasAgent(
        obs_dim=batch_loader.obs_dim,
        act_dim=batch_loader.act_dim,
        dynamics=dynamics,
        tau=tau,
        beta=beta,
        alpha=alpha,
        lmbda=lmbda,
        gamma=gamma,
        max_action=max_action,
        policy_freq=policy_freq,
        device=device,
    )
    for step in range(1, agent_steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)
        if log_interval > 0 and step % log_interval == 0:
            print(f"[agent][train] step={step}/{agent_steps}")
        if eval_interval > 0 and step % eval_interval == 0:
            loss_q_sum = 0.0
            loss_td3_sum = 0.0
            loss_corr_sum = 0.0
            loss_pi_sum = 0.0
            for _ in range(eval_batches):
                e_batch = batch_loader.sample_batch(batch_size)
                s, a, r, sn, d = _as_tensors(e_batch, device)
                with torch.no_grad():
                    loss_q_sum += float(agent.loss_critic(s, a, r, sn, d).item())
                    loss_td3_sum += float(agent.loss_td3(s).item())
                    loss_corr_sum += float(agent.loss_correction(s, sn).item())
                    loss_pi_sum += float(agent.loss_actor(s, sn).item())
            print(
                f"[agent][eval] step={step}/{agent_steps} "
                f"loss_q={loss_q_sum / eval_batches:.6f} "
                f"loss_td3={loss_td3_sum / eval_batches:.6f} "
                f"loss_correction={loss_corr_sum / eval_batches:.6f} "
                f"loss_pi={loss_pi_sum / eval_batches:.6f}"
            )
            _append_csv(
                agent_csv,
                [
                    step,
                    loss_q_sum / eval_batches,
                    loss_td3_sum / eval_batches,
                    loss_corr_sum / eval_batches,
                    loss_pi_sum / eval_batches,
                ],
                ["step", "loss_q", "loss_td3", "loss_correction", "loss_pi"],
            )


def main() -> None:
    dataset = get_dataset("halfcheetah_medium")
    batch_loader = MinariLoader(dataset=dataset, seed=42)
    print_stage("Load")
    train(
        batch_loader=batch_loader,
        log_interval=10_000,
        eval_interval=20_000,
        eval_batches=4,
    )
    print_stage("Done")


if __name__ == "__main__":
    main()
