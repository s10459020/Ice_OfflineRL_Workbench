from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ice_offline.agent._spec import Agent
from ice_offline.agent._spec import MetricValues
from ice_offline.agent.sac import SACAgent
from ice_offline.dataset._types import Batch


class _DynamicsMember(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, obs_size)
        self.logvar_head = torch.nn.Linear(256, obs_size)

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(torch.cat([o, a], dim=1))
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden).clamp(-10.0, 2.0)
        return mean, logvar


class _DynamicsModel(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, ensemble_size: int):
        super().__init__()
        self.members = torch.nn.ModuleList(
            [_DynamicsMember(obs_size, act_size) for _ in range(ensemble_size)]
        )

    def loss(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        losses = []
        for member in self.members:
            mean, logvar = member(o, a)
            inv_var = torch.exp(-logvar)
            nll = 0.5 * (logvar + (on - mean).pow(2) * inv_var)
            losses.append(nll.mean())
        return torch.stack(losses).mean()

    def sample(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        outputs = []
        for member in self.members:
            mean, logvar = member(o, a)
            std = torch.exp(0.5 * logvar)
            outputs.append(mean + std * torch.randn_like(std))
        stacked = torch.stack(outputs, dim=1)
        return stacked.mean(dim=1)


class _TransitionEncoder(torch.nn.Module):
    def __init__(self, obs_size: int, latent_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, latent_size)
        self.logvar_head = torch.nn.Linear(256, latent_size)

    def forward(self, o: torch.Tensor, on: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(torch.cat([o, on], dim=1))
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden).clamp(-10.0, 10.0)
        return mean, logvar


class _TransitionDecoder(torch.nn.Module):
    def __init__(self, obs_size: int, latent_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + latent_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def forward(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([o, z], dim=1))


class _TransitionModel(torch.nn.Module):
    def __init__(self, obs_size: int, latent_size: int):
        super().__init__()
        self.obs_size = obs_size
        self.latent_size = latent_size
        self.encoder = _TransitionEncoder(obs_size, latent_size)
        self.decoder = _TransitionDecoder(obs_size, latent_size)

    def loss(self, o: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encoder(o, on)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        reconstruction = self.decoder(o, z)
        loss_reconstruction = F.mse_loss(reconstruction, on)
        loss_kl = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).mean()
        return loss_reconstruction + loss_kl

    def sample(self, o: torch.Tensor, n_samples: int) -> torch.Tensor:
        batch_size = o.shape[0]
        o_repeated = o.repeat_interleave(n_samples, dim=0)
        z = torch.randn(batch_size * n_samples, self.latent_size, device=o.device)
        on = self.decoder(o_repeated, z)
        return on.view(batch_size, n_samples, self.obs_size)


@dataclass
class SDCModel(Agent):
    obs_size: int
    act_size: int
    state_transition_latent_size: int = 8
    dynamics_ensemble_size: int = 5
    dynamics_learning_rate: float = 3e-4
    transition_learning_rate: float = 3e-4
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.dynamics = _DynamicsModel(
            self.obs_size,
            self.act_size,
            self.dynamics_ensemble_size,
        ).to(self.device)
        self.transition = _TransitionModel(
            self.obs_size,
            self.state_transition_latent_size,
        ).to(self.device)
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=self.dynamics_learning_rate,
        )
        self.transition_optimizer = torch.optim.Adam(
            self.transition.parameters(),
            lr=self.transition_learning_rate,
        )

    def prepare(self) -> "SDCModel":
        self.dynamics.eval()
        self.transition.eval()
        for parameter in self.dynamics.parameters():
            parameter.requires_grad = False
        for parameter in self.transition.parameters():
            parameter.requires_grad = False
        return self

    def update(self, batch: Batch) -> None:
        self.dynamics_optimizer.zero_grad()
        loss_dynamics = self.loss_dynamics(batch)
        loss_dynamics.backward()
        self.dynamics_optimizer.step()

        self.transition_optimizer.zero_grad()
        loss_transition = self.loss_transition(batch)
        loss_transition.backward()
        self.transition_optimizer.step()

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        loss_dynamics = self.loss_dynamics(batch)
        grad_dynamics = self._grad_norm(loss_dynamics, self.dynamics.parameters())
        self.dynamics_optimizer.zero_grad()
        loss_dynamics.backward()
        self.dynamics_optimizer.step()

        loss_transition = self.loss_transition(batch)
        grad_transition = self._grad_norm(loss_transition, self.transition.parameters())
        self.transition_optimizer.zero_grad()
        loss_transition.backward()
        self.transition_optimizer.step()
        return {
            "loss_dynamics": loss_dynamics.detach(),
            "grad_dynamics": grad_dynamics.detach(),
            "loss_transition": loss_transition.detach(),
            "grad_transition": grad_transition.detach(),
        }

    def _save_dict(self) -> dict[str, object]:
        return {
            "dynamics": self.dynamics.state_dict(),
            "transition": self.transition.state_dict(),
            "dynamics_optimizer": self.dynamics_optimizer.state_dict(),
            "transition_optimizer": self.transition_optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, object]) -> None:
        self.dynamics.load_state_dict(state["dynamics"])
        self.transition.load_state_dict(state["transition"])
        self.dynamics_optimizer.load_state_dict(state["dynamics_optimizer"])
        self.transition_optimizer.load_state_dict(state["transition_optimizer"])

    def loss_dynamics(self, batch: Batch) -> torch.Tensor:
        o, a, _, on, _ = batch
        return self.dynamics.loss(o, a, on)

    def loss_transition(self, batch: Batch) -> torch.Tensor:
        o, _, _, on, _ = batch
        return self.transition.loss(o, on)


@dataclass
class SDCAgent(SACAgent):
    model: SDCModel | None = None
    scale_noise: float = 0.1
    weight_penalty: float = 1.0
    threshold_penalty: float = 0.05
    count_sample: int = 4
    mmd_sigma: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model is None:
            raise ValueError("SDCAgent requires a pretrained SDCModel.")
        self.model = self.model.prepare()

    def _save_dict(self) -> dict[str, object]:
        state = super()._save_dict()
        state["sdc_model"] = self.model._save_dict()
        return state

    def _load_dict(self, state: dict[str, object]) -> None:
        super()._load_dict(state)
        self.model._load_dict(state["sdc_model"])
        self.model.prepare()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        loss_sdc = self.loss_state_deviation(batch)
        return self.loss_sac(batch) + self.weight_penalty * (
            loss_sdc - self.threshold_penalty
        )

    def loss_state_deviation(self, batch: Batch) -> torch.Tensor:
        o, _, _, _, _ = batch
        batch_size = o.shape[0]
        o_noisy = o[:, None, :] + self.scale_noise * torch.randn(
            batch_size,
            self.count_sample,
            self.obs_size,
            device=o.device,
        )
        o_flat = o_noisy.reshape(batch_size * self.count_sample, self.obs_size)
        a_flat, _ = self.actor.sample(o_flat)
        dynamics_next = self.model.dynamics.sample(o_flat, a_flat)

        with torch.no_grad():
            transition_next = self.model.transition.sample(o, self.count_sample).reshape(
                batch_size * self.count_sample,
                self.obs_size,
            )
        return self.mmd(dynamics_next, transition_next)

    def mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        kernel_xx = self.gaussian_kernel(x, x).mean()
        kernel_yy = self.gaussian_kernel(y, y).mean()
        kernel_xy = self.gaussian_kernel(x, y).mean()
        return kernel_xx + kernel_yy - 2.0 * kernel_xy

    def gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(x, y).pow(2)
        return torch.exp(-distance / (2.0 * self.mmd_sigma**2))
