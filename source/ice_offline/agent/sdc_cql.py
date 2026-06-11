from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from ice_offline.agent.cql import CQLAgent
from ice_offline.dataset._types import Batch


class _DynamicsModel(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([o, a], dim=1))


class _TransitionModel(torch.nn.Module):
    def __init__(self, obs_size: int, noise_size: int):
        super().__init__()
        self.obs_size = obs_size
        self.noise_size = noise_size
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + noise_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def forward(self, o: torch.Tensor, n_samples: int) -> torch.Tensor:
        batch = o.shape[0]
        o_rep = o.repeat_interleave(n_samples, dim=0)
        noise = torch.randn(batch * n_samples, self.noise_size, device=o.device)
        on = self.network(torch.cat([o_rep, noise], dim=1))
        return on.view(batch, n_samples, self.obs_size)


@dataclass
class SDCCQLAgent(CQLAgent):
    agent_name: ClassVar[str] = "sdc_cql"
    state_noise_beta: float = 0.1
    state_transition_noise_size: int = 8
    sdc_weight: float = 1.0
    sdc_threshold: float = 0.05
    sdc_samples: int = 4
    mmd_sigma: float = 10.0
    dynamics_learning_rate: float = 3e-4
    transition_learning_rate: float = 3e-4

    def __post_init__(self):
        super().__post_init__()
        self.dynamics = _DynamicsModel(self.obs_size, self.act_size).to(self.device)
        self.transition = _TransitionModel(self.obs_size, self.state_transition_noise_size).to(self.device)
        self.state_models_optimizer = torch.optim.Adam(
            [
                {"params": self.dynamics.parameters(), "lr": self.dynamics_learning_rate},
                {"params": self.transition.parameters(), "lr": self.transition_learning_rate},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    def update(self, batch: Batch):
        self.update_state_models(batch)
        self.update_critic(batch)
        self.update_actor(batch)
        self.critic.update_target_soft()

    def update_state_models(self, batch: Batch) -> None:
        self.state_models_optimizer.zero_grad()
        model_loss = self.loss_state_models(batch)
        model_loss.backward()
        self.state_models_optimizer.step()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        state = super()._save_dict()
        state.update(
            {
                "dynamics": self.dynamics.state_dict(),
                "transition": self.transition.state_dict(),
                "state_models_optimizer": self.state_models_optimizer.state_dict(),
            }
        )
        return state

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        super()._load_dict(state)
        self.dynamics.load_state_dict(state["dynamics"])
        self.transition.load_state_dict(state["transition"])
        self.state_models_optimizer.load_state_dict(state["state_models_optimizer"])

    def loss_state_models(self, batch: Batch) -> torch.Tensor:
        o, a, _, on, _ = batch
        dynamics_loss = F.mse_loss(self.dynamics(o, a), on)
        transition_loss = F.mse_loss(self.transition(o, 1).squeeze(1), on)
        return dynamics_loss + transition_loss

    def loss_dynamics(self, batch: Batch) -> torch.Tensor:
        o, a, _, on, _ = batch
        return F.mse_loss(self.dynamics(o, a), on)

    def loss_transition(self, batch: Batch) -> torch.Tensor:
        o, _, _, on, _ = batch
        return F.mse_loss(self.transition(o, 1).squeeze(1), on)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        o, _, _, _, _ = batch
        a, _ = self.actor.sample(o)
        q_t = self.critic.q_min(o, a)
        sdc = self.loss_state_deviation(batch)
        return -q_t.mean() + self.sdc_weight * (sdc - self.sdc_threshold)

    def loss_state_deviation(self, batch: Batch) -> torch.Tensor:
        o, _, _, _, _ = batch
        batch = o.shape[0]
        o_noisy = o[:, None, :] + self.state_noise_beta * torch.randn(
            batch, self.sdc_samples, self.obs_size, device=o.device
        )
        o_flat = o_noisy.reshape(batch * self.sdc_samples, self.obs_size)
        a_flat, _ = self.actor.sample(o_flat)
        dynamics_next = self.dynamics(o_flat, a_flat).view(batch * self.sdc_samples, self.obs_size)

        with torch.no_grad():
            transition_next = self.transition(o, self.sdc_samples).reshape(
                batch * self.sdc_samples, self.obs_size
            )
        return self.mmd(dynamics_next, transition_next)

    def mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.gaussian_kernel(x, x).mean()
        k_yy = self.gaussian_kernel(y, y).mean()
        k_xy = self.gaussian_kernel(x, y).mean()
        return k_xx + k_yy - 2.0 * k_xy

    def gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, y).pow(2)
        return torch.exp(-dist / (2.0 * self.mmd_sigma**2))

