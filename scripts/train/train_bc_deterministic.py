import numpy as np
import torch

from ice_offline.agent.bc_continuous_deterministic import BCAgentContinuousDeterministic
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.run.evaluator2 import Evaluator2
from ice_offline.tools.printer import print_stage


def eval_loss_pi(agent: BCAgentContinuousDeterministic, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, _, _, _ = episode_batch
    with torch.no_grad():
        return {"loss_pi": float(agent.loss_actor(o, a).item())}


def eval_reward(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"reward_sum": float(r.sum().item())}


def train(
    task_id: str,
    batch_loader: MinariLoader,
    *,
    seed: int = 42,
    batch_size: int = 64,
    steps: int = 100_000,
    log_interval: int = 0,
    eval_interval: int = 2_000,
    eval_offline_n: int = 8,
    eval_online_n: int = 3,
    eval_env_fn=None,
    recode_eval: bool = True,
    recode_reset: bool = True,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    print_stage("Train BC Deterministic")
    agent = BCAgentContinuousDeterministic(
        obs_size=batch_loader.obs_dim,
        act_size=batch_loader.act_dim,
    )

    evaluator = Evaluator2(
        runner_id=task_id,
        eval_interval=eval_interval,
        eval_offline_n=eval_offline_n,
        eval_online_n=eval_online_n,
        eval_offline_fns=[eval_loss_pi],
        eval_online_fns=[eval_reward],
        recode_eval=recode_eval,
        recode_reset=recode_reset,
    )

    for step in range(1, steps + 1):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)
        if log_interval > 0 and step % log_interval == 0:
            print(f"[bc_deterministic][step] {step}")
        evaluator.eval_offline(step=step, agent=agent, batch_loader=batch_loader, batch_size=batch_size)
        evaluator.eval_online(step=step, agent=agent, env_fn=eval_env_fn)
        evaluator.print(step)
        evaluator.recode(step)


def main() -> None:
    task_id = "invertedpendulum_expert__bc_deterministic"
    dataset = get_dataset("invertedpendulum_expert")
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
