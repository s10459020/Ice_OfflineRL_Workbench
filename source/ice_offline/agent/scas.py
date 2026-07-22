import torch
import torch.nn.functional as F

from ice_offline.agent._spec import MetricValues
from ice_offline.agent._spec import Agent
from ice_offline.agent.td3 import _Pi
from ice_offline.agent.td3 import _Q
from ice_offline.agent.td3 import TD3Actor
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3 import TD3Critic
from ice_offline.dataset._types import Batch

class _M(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a], -1)
        return self.network(x)

class ScasDynamic(Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.noise_scale = config.get("noise_scale", 3e-3)
        self.model = _M(self.obs_size, self.act_size, config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    # ====================
    # extend 
    # ====================
    def prepare(self) -> "ScasDynamic":
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        return self

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.model(o, a)

    def noise_state(self, o: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(o.shape, device=o.device) * self.noise_scale
        return o + noise

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_dynamic",
            "grad_dynamic",
        ]

    def update(self, batch: Batch) -> MetricValues:
        loss_dynamic, metrics = self.loss_dynamic(batch)
        self.optimizer.zero_grad()
        loss_dynamic.backward()
        self.optimizer.step()
        return metrics

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
    def loss_dynamic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s, a, _, sn, _ = batch
        # loss: E_{s,a,s'~D} [||M(s,a) - s'||^2]
        pred = self.model(s, a)
        loss = F.mse_loss(pred, sn)
        return loss, {
            "loss_dynamic": self._value(loss.detach()),
            "grad_dynamic": self._grad_norm(loss, self.model.parameters()),
        }
        

class ScasAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, dynamics: ScasDynamic, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_correction = config.get("weight_correction", 0.25)
        self.scale_gap = config.get("scale_gap", 5.0)
        self.max_gap = config.get("max_gap", 50.0)
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.actor = TD3Actor(self.obs_size, self.act_size, config=config, pi_cls=_Pi).to(self.device)
        self.critic = TD3Critic(self.obs_size, self.act_size, config=config, q_cls=_Q).to(self.device)
        self.dynamics = dynamics.prepare()
        self.actor_optimizer = torch.optim.Adam(self.actor.param_actor())
        self.critic_optimizer = torch.optim.Adam(self.critic.param_critic())

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_td3",
            "grad_td3",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "target_q",
        ]

    def update(self, batch: Batch) -> MetricValues:
        self.update_step += 1
        metrics = self.update_critic(batch)

        if self.update_step % self.update_actor_interval == 0:
            metrics |= self.update_actor(batch)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

        return metrics

    # ====================
    # Actor loss
    # ====================
    def loss_correction(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # loss = E_{s,s'~D, ps~perturbed(s)} [exp( scale * ( V' - V ) ) * ||M(ps, pi(ps)) - s'||^2]
        s, _, _, sn, _ = batch
        a = self.actor.pi(s)
        v = self.critic.q_min(s, a) # scas V(s) = Q(s, pi(s))
        an = self.actor.pi(sn)
        vn = self.critic.q_min(sn, an) # scas V(s') = Q(s', pi(s'))

        weight = (
            self.scale_gap * (vn.detach() - v.detach())
        ).exp().clamp(max=self.max_gap)

        ps = self.dynamics.noise_state(s)
        pa = self.actor.pi(ps)
        mse_M = (self.dynamics.forward(ps, pa) - sn) ** 2
        loss = (weight * mse_M).mean()
        return loss, {
            "loss_correction": self._value(loss.detach()),
            "grad_correction": self._grad_norm(loss, self.actor.param_actor()),
        }

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td3, metrics_td3 = self.loss_td3(batch)
        loss_correction, metrics_correction = self.loss_correction(batch)
        loss = (
            (1.0 - self.weight_correction) * loss_td3
            + self.weight_correction * loss_correction
        )
        return loss, metrics_td3 | metrics_correction | {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        }
