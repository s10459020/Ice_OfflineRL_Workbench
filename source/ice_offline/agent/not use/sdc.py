import torch
import torch.nn.functional as F

from ice_offline.agent._spec import Agent
from ice_offline.agent._spec import MetricValues
from ice_offline.agent.sac import SACAgent
from ice_offline.dataset._types import Batch


class GaussianMember(torch.nn.Module):
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


class EnsembleDynamics(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.members = torch.nn.ModuleList(
            [GaussianMember(obs_size, act_size) for _ in range(config.get("dynamics_ensemble_size", 5))]
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


class CvaeTransition(torch.nn.Module):
    def __init__(self, obs_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.obs_size = obs_size
        self.latent_size = config.get("latent_size", 8)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_size * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.encoder_mean = torch.nn.Linear(256, self.latent_size)
        self.encoder_logvar = torch.nn.Linear(256, self.latent_size)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(obs_size + self.latent_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def encode(self, o: torch.Tensor, on: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(torch.cat([o, on], dim=1))
        mean = self.encoder_mean(hidden)
        logvar = self.encoder_logvar(hidden).clamp(-10.0, 10.0)
        return mean, logvar

    def decode(self, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([o, z], dim=1))

    def loss(self, o: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode(o, on)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        reconstruction = self.decode(o, z)
        loss_reconstruction = F.mse_loss(reconstruction, on)
        loss_kl = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).mean()
        return loss_reconstruction + loss_kl

    def sample(self, o: torch.Tensor, n_samples: int) -> torch.Tensor:
        batch_size = o.shape[0]
        o_repeated = o.repeat_interleave(n_samples, dim=0)
        z = torch.randn(batch_size * n_samples, self.latent_size, device=o.device)
        on = self.decode(o_repeated, z)
        return on.view(batch_size, n_samples, self.obs_size)


class SDCModel(Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.dynamics = EnsembleDynamics(self.obs_size, self.act_size, config).to(self.device)
        self.transition = CvaeTransition(self.obs_size, config).to(self.device)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters())
        self.transition_optimizer = torch.optim.Adam(self.transition.parameters())

    def prepare(self) -> "SDCModel":
        self.dynamics.eval()
        self.transition.eval()
        for parameter in self.dynamics.parameters():
            parameter.requires_grad = False
        for parameter in self.transition.parameters():
            parameter.requires_grad = False
        return self

    def update(self, batch: Batch) -> MetricValues:
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


class SDCAgent(SACAgent):
    def __init__(self, obs_size: int, act_size: int, model: SDCModel | None = None, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.model = model
        self.scale_noise = config.get("scale_noise", 0.1)
        self.weight_penalty = config.get("weight_penalty", 1.0)
        self.threshold_penalty = config.get("threshold_penalty", 0.05)
        self.count_sample = config.get("count_sample", 4)
        self.mmd_sigma = config.get("mmd_sigma", 10.0)
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
