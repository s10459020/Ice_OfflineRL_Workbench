from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ice_offline.agent._spec import Agent
from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.sdc_cql import _DynamicsModel
from ice_offline.agent.sdc_cql import _TransitionModel
from ice_offline.dataset._types import Batch


@dataclass
class SDCPreModel(Agent):
    obs_size: int
    act_size: int
    state_transition_noise_size: int = 8
    dynamics_learning_rate: float = 3e-4
    transition_learning_rate: float = 3e-4
    device: str = "cpu"

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        self.dynamics = _DynamicsModel(self.obs_size, self.act_size).to(self.device)
        self.transition = _TransitionModel(self.obs_size, self.state_transition_noise_size).to(self.device)
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=self.dynamics_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.transition_optimizer = torch.optim.Adam(
            self.transition.parameters(),
            lr=self.transition_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Extend
    # ====================
    def prepare(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        # The pretrained models are fixed during SDC actor optimization.
        self.dynamics.eval()
        self.transition.eval()
        for p in self.dynamics.parameters():
            p.requires_grad = False
        for p in self.transition.parameters():
            p.requires_grad = False
        return self.dynamics, self.transition

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch):
        self.dynamics_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss = self.loss_state_models(batch)
        loss.backward()
        self.dynamics_optimizer.step()
        self.transition_optimizer.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "dynamics": self.dynamics.state_dict(),
            "transition": self.transition.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.dynamics.load_state_dict(state["dynamics"])
        self.transition.load_state_dict(state["transition"])

    # ====================
    # State model loss
    # ====================
    def loss_state_models(self, batch: Batch) -> torch.Tensor:
        return self.loss_dynamics(batch) + self.loss_transition(batch)

    def loss_dynamics(self, batch: Batch) -> torch.Tensor:
        # loss = E{s,a,s'~D}[ ||M(s,a)-s'||^2 ]
        o, a, _, on, _ = batch
        return F.mse_loss(self.dynamics(o, a), on)

    def loss_transition(self, batch: Batch) -> torch.Tensor:
        # loss = E{s,s'~D,z~N}[ ||U(s,z)-s'||^2 ]
        o, _, _, on, _ = batch
        return F.mse_loss(self.transition(o, 1).squeeze(1), on)


@dataclass
class SDCPreAgent(CQLAgent):
    state_models: SDCPreModel | None = None
    state_transition_noise_size: int = 8
    state_noise_beta: float = 0.1
    sdc_weight: float = 1.0
    sdc_threshold: float = 0.05
    sdc_samples: int = 4
    mmd_sigma: float = 10.0

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state_models is None:
            self.dynamics = _DynamicsModel(self.obs_size, self.act_size).to(self.device)
            self.transition = _TransitionModel(self.obs_size, self.state_transition_noise_size).to(self.device)
        else:
            self.dynamics, self.transition = self.state_models.prepare()
            self.dynamics = self.dynamics.to(self.device)
            self.transition = self.transition.to(self.device)

        self.dynamics.eval()
        self.transition.eval()
        for p in self.dynamics.parameters():
            p.requires_grad = False
        for p in self.transition.parameters():
            p.requires_grad = False

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, object]:
        state = super()._save_dict()
        state["dynamics"] = self.dynamics.state_dict()
        state["transition"] = self.transition.state_dict()
        return state

    def _load_dict(self, state: dict[str, object]) -> None:
        super()._load_dict(state)
        self.dynamics.load_state_dict(state["dynamics"])
        self.transition.load_state_dict(state["transition"])

    # ====================
    # Actor loss
    # ====================
    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # loss = -E{s~D,a~pi}[Q(s,a)] + weight * (SDC(s)-threshold)
        o, _, _, _, _ = batch
        a, _ = self.actor.sample(o)
        q = self.critic.q_min(o, a)
        sdc = self.loss_state_deviation(o)
        return -q.mean() + self.sdc_weight * (sdc - self.sdc_threshold)

    def loss_state_deviation(self, o: torch.Tensor) -> torch.Tensor:
        # SDC(s) = MMD(M(s+noise, pi(s+noise)), U(s,z))
        batch_size = o.shape[0]
        o_noisy = o[:, None, :] + self.state_noise_beta * torch.randn(
            batch_size,
            self.sdc_samples,
            self.obs_size,
            device=o.device,
        )
        o_flat = o_noisy.reshape(batch_size * self.sdc_samples, self.obs_size)
        a_flat, _ = self.actor.sample(o_flat)
        dynamics_next = self.dynamics(o_flat, a_flat).view(
            batch_size * self.sdc_samples,
            self.obs_size,
        )

        with torch.no_grad():
            transition_next = self.transition(o, self.sdc_samples).reshape(
                batch_size * self.sdc_samples,
                self.obs_size,
            )
        return self.mmd(dynamics_next, transition_next)

    def mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # MMD^2(x,y) = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
        k_xx = self.gaussian_kernel(x, x).mean()
        k_yy = self.gaussian_kernel(y, y).mean()
        k_xy = self.gaussian_kernel(x, y).mean()
        return k_xx + k_yy - 2.0 * k_xy

    def gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, y).pow(2)
        return torch.exp(-dist / (2.0 * self.mmd_sigma**2))
