import numpy as np
import torch

from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.scas import ScasDynamic
from ice_offline.dataset._lookup import get_dataset
from ice_offline.pipeline.batch_loader import MinariLoader
from ice_offline.tools.printer import print_stage


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
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    print_stage("Train Dynamics")
    dynamics = ScasDynamic(
        obs_dim=batch_loader.obs_dim,
        act_dim=batch_loader.act_dim,
        learning_rate=1e-3,
        device=device,
    )
    for _ in range(dynamics_steps):
        batch = batch_loader.sample_batch(batch_size)
        dynamics.update(batch)

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
    for _ in range(agent_steps):
        batch = batch_loader.sample_batch(batch_size)
        agent.update(batch)


def main() -> None:
    dataset = get_dataset("halfcheetah_medium")
    batch_loader = MinariLoader(dataset=dataset, seed=42)
    print_stage("Load")
    train(batch_loader=batch_loader)
    print_stage("Done")


if __name__ == "__main__":
    main()
