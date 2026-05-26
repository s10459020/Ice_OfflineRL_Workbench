import numpy as np
import torch

from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.run.evaluator2 import Evaluator2
from ice_offline.tools.printer import print_stage


def eval_loss_dynamic(dynamics: ScasDynamic, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    s, a, _, sn, _ = episode_batch
    with torch.no_grad():
        return {"loss_dynamic": float(dynamics.loss_dynamic(s, a, sn).item())}


def eval_loss_agent(agent: ScasAgent, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    s, a, r, sn, d = episode_batch
    with torch.no_grad():
        return {
            "loss_q": float(agent.loss_critic(s, a, r, sn, d).item()),
            "loss_td3": float(agent.loss_td3(s).item()),
            "loss_correction": float(agent.loss_correction(s, sn).item()),
            "loss_pi": float(agent.loss_actor(s, sn).item()),
        }


def eval_return(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"return": float(r.sum().item())}


def train(
    task_id: str,
    batch_loader: MinariLoader,
    *,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 256,
    dynamics_steps: int = 100_000,
    agent_steps: int = 500_000,
    max_action: float = 1.0,
    tau: float = 0.005,
    gamma: float = 0.99,
    beta: float = 3e-3,
    alpha: float = 5.0,
    lmbda: float = 0.25,
    policy_freq: int = 2,
    log_interval: int = 0,
    eval_interval: int = 1000,
    eval_offline_n: int = 8,
    eval_online_n: int = 4,
    eval_env_fn=None,
    recode_eval: bool = True,
    recode_reset: bool = True,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    dynamics_evaluator = Evaluator2(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_offline_fns=[eval_loss_dynamic],
        recode_eval=recode_eval,
        recode_reset=recode_reset,
    )



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
            print(f"[dynamics][step] {step}")
        dynamics_evaluator.eval_offline(step=step, agent=dynamics, batch_loader=batch_loader, batch_size=batch_size)
        dynamics_evaluator.print(step)
        dynamics_evaluator.recode(step)



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
    agent_evaluator = Evaluator2(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss_agent],
        eval_online_fns=[eval_return],
        recode_eval=recode_eval,
        recode_reset=recode_reset,
    )

    for step in range(1, agent_steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)
        if log_interval > 0 and step % log_interval == 0:
            print(f"[agent][step] {step}")
        agent_evaluator.eval_offline(step=step, agent=agent, batch_loader=batch_loader, batch_size=batch_size)
        agent_evaluator.eval_online(step=step, agent=agent, env_fn=eval_env_fn)
        agent_evaluator.print(step)
        agent_evaluator.recode(step)


def main() -> None:
    TASK_ID = "halfcheetah_medium__scas"
    dataset = get_dataset("halfcheetah_medium")
    batch_loader = MinariLoader(dataset=dataset, seed=42)
    print_stage("Load")
    train(
        task_id=TASK_ID,
        batch_loader=batch_loader,
        eval_env_fn=dataset.make_eval_env,
    )
    print_stage("Done")


if __name__ == "__main__":
    main()
