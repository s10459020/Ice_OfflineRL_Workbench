from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.cql import _Adam
from ice_offline.dataset._spec import TorchBuffer


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
        self.dynamics_optim = _Adam(self.dynamics_learning_rate)(self.dynamics.parameters())
        self.transition_optim = _Adam(self.transition_learning_rate)(self.transition.parameters())

    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list
        r = batch.rew_list.view(-1, 1)
        on = batch.next_obs_list
        d = batch.done_list.view(-1, 1)

        model_loss = self.loss_state_models(o, a, on)
        self.dynamics_optim.zero_grad()
        self.transition_optim.zero_grad()
        model_loss.backward()
        self.dynamics_optim.step()
        self.transition_optim.step()

        critic_loss = self.loss_critic(o, a, r, on, d, update_alpha=True)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = self.loss_actor(o)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic.update_target_soft()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        state = super()._save_dict()
        state.update(
            {
                "dynamics": self.dynamics.state_dict(),
                "transition": self.transition.state_dict(),
                "dynamics_optimizer": self.dynamics_optim.state_dict(),
                "transition_optimizer": self.transition_optim.state_dict(),
            }
        )
        return state

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        super()._load_dict(state)
        self.dynamics.load_state_dict(state["dynamics"])
        self.transition.load_state_dict(state["transition"])
        self.dynamics_optim.load_state_dict(state["dynamics_optimizer"])
        self.transition_optim.load_state_dict(state["transition_optimizer"])

    def loss_state_models(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        dynamics_loss = F.mse_loss(self.dynamics(o, a), on)
        transition_loss = F.mse_loss(self.transition(o, 1).squeeze(1), on)
        return dynamics_loss + transition_loss

    def loss_dynamics(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.dynamics(o, a), on)

    def loss_transition(self, o: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.transition(o, 1).squeeze(1), on)

    def loss_actor(self, o: torch.Tensor, update_alpha: bool = True) -> torch.Tensor:
        a, _ = self.actor.sample(o)
        q_t = self.critic.qq_min(o, a)
        sdc = self.loss_state_deviation(o)
        return -q_t.mean() + self.sdc_weight * (sdc - self.sdc_threshold)

    def loss_state_deviation(self, o: torch.Tensor) -> torch.Tensor:
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
