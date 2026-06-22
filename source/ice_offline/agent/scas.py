from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ice_offline.agent._spec import MetricValues
from ice_offline.agent._spec import Agent
from ice_offline.agent.td3 import TD3Actor
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3 import TD3Critic
from ice_offline.dataset._types import Batch

class _M(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, noise_scale: float = 3e-3):
        super().__init__()
        self.noise_scale = noise_scale
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a], -1)
        return self.network(x)

    def noise_state(self, o: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(o.shape, device=o.device) * self.noise_scale
        return o + noise

@dataclass
class ScasDynamic(Agent):
    obs_size: int
    act_size: int
    learning_rate: float = 1e-3
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.model = _M(self.obs_size, self.act_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    # ====================
    # extend 
    # ====================
    def prepare(self) -> torch.nn.Module:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        return self.model
    
    def update(self, batch: Batch):
        self.optimizer.zero_grad()
        loss = self.loss_dynamic(batch)
        loss.backward() 
        self.optimizer.step()

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        loss_dynamic = self.loss_dynamic(batch)
        grad_dynamic = self._grad_norm(loss_dynamic, self.model.parameters())
        self.optimizer.zero_grad()
        loss_dynamic.backward()
        self.optimizer.step()
        return {
            "loss_dynamic": loss_dynamic.detach(),
            "grad_dynamic": grad_dynamic.detach(),
        }

    # ====================
    # extend 
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {"model": self.model.state_dict()}

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state["model"])
  

    # ====================
    # mathmatics
    # ====================
    def loss_dynamic(self, batch: Batch) -> torch.Tensor:
        s, a, _, sn, _ = batch
        # loss: E_{s,a,s'~D} [||M(s,a) - s'||^2]
        pred = self.model(s, a)
        return F.mse_loss(pred, sn)
        

@dataclass
class ScasAgent(TD3Agent):
    dynamics: ScasDynamic | None = None
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 3e-4
    alpha: float = 5.0
    lmbda: float = 0.25
    q_count: int = 4
    max_weight: float = 50.0

    def __post_init__(self) -> None:
        self.actor = TD3Actor(self.obs_size, self.act_size).to(self.device)
        self.critic = TD3Critic(self.obs_size, self.act_size, q_count=self.q_count).to(self.device)
        self.dynamics.prepare()
        self.dynamics.model = self.dynamics.model.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.pi.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.q_networks.parameters(),
            lr=self.critic_learning_rate,
        )

    # ====================
    # Actor loss
    # ====================    
    def loss_correction(self, batch: Batch) -> torch.Tensor:
        # loss = E_{s,s'~D, ps~perturbed(s)} [exp( alpha* ( V' - V ) ) * ||M(ps,a) - s'||^2]
        s, _, _, sn, _ = batch
        a = self.actor.pi(s)
        v = self.critic.q_mean(s, a) # scas V(s) = Q(s, pi(s))
        an = self.actor.pi(sn)
        vn = self.critic.q_mean(sn, an) # scas V(s') = Q(s', pi(s'))

        weight = (
            self.alpha * (vn.detach() - v.detach())
        ).exp().clamp(max = self.max_weight)

        ps = self.dynamics.noise_state(s)
        pa = self.actor.pi(ps)
        mse_M = (self.dynamics.model(ps, pa) - sn) ** 2
        return (weight * mse_M).mean() # mean over batch

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        return (1.0 - self.lmbda) * self.loss_td3(batch) + self.lmbda * self.loss_correction(batch)
