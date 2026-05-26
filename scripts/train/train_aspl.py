import numpy as np
import torch

from ice_offline.agent.aspl import AsplAgent
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.run.evaluator2 import Evaluator2
from ice_offline.tools.printer import print_stage


def eval_loss_aspl(agent: AsplAgent, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    s, a, r, sn, d = episode_batch
    with torch.no_grad():
        q_target = agent._td_target(sn, r, d)
        return {
            "loss_critic": float(agent.loss_critic(s, a, r, sn, d).item()),
            "loss_td": float(agent.loss_td_with_target(s, a, q_target).item()),
            "loss_punish": float(agent.loss_punish_with_target(s, a, q_target).item()),
            "loss_actor": float(agent.loss_td3_variant(s).item()),
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
    steps: int = 500_000,
    max_action: float = 1.0,
    tau: float = 0.005,
    gamma: float = 0.99,
    beta: float = 3e-3,
    alpha: float = 2.5,
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

    print_stage("Train ASPL")
    agent = AsplAgent(
        obs_dim=batch_loader.obs_dim,
        act_dim=batch_loader.act_dim,
        max_action=max_action,
        tau=tau,
        beta=beta,
        alpha=alpha,
        lmbda=lmbda,
        gamma=gamma,
        policy_freq=policy_freq,
        device=device,
    )

    evaluator = Evaluator2(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss_aspl],
        eval_online_fns=[eval_return],
        recode_eval=recode_eval,
        recode_reset=recode_reset,
    )

    for step in range(1, steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)
        if log_interval > 0 and step % log_interval == 0:
            print(f"[aspl][step] {step}")
        evaluator.eval_offline(step=step, agent=agent, batch_loader=batch_loader, batch_size=batch_size)
        evaluator.eval_online(step=step, agent=agent, env_fn=eval_env_fn)
        evaluator.print(step)
        evaluator.recode(step)


def main() -> None:
    task_id = "halfcheetah_medium__aspl"
    dataset = get_dataset("halfcheetah_medium")
    batch_loader = MinariLoader(dataset=dataset, seed=42)
    print_stage("Load")
    train(
        task_id=task_id,
        batch_loader=batch_loader,
        eval_env_fn=dataset.make_eval_env,
    )
    print_stage("Done")


if __name__ == "__main__":
    main()





