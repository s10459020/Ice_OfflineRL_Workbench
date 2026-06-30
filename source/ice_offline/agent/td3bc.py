import torch

from ice_offline.agent.td3 import TD3Agent
from ice_offline.dataset._types import Batch


class TD3BCAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_td3 = config.get("weight_td3", 2.5)
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_normal",
            "grad_normal",
            "loss_bc",
            "grad_bc",
            "loss_actor",
            "grad_actor",
            "target_q",
        ]

    # ====================
    # Actor loss
    # ====================
    def loss_normal(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, _, _, _, _ = batch
        a = self.actor.pi(o)
        q = self.critic.q_min(o, a)
        loss = -q.mean() / q.abs().mean().detach()
        return loss, {
            "loss_normal": self._value(loss.detach()),
            "grad_normal": self._grad_norm(loss, self.actor.param_actor()),
        }

    def loss_bc(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, a, _, _, _ = batch
        a_pred = self.actor.pi(o)
        loss = ((a - a_pred) ** 2).mean()
        return loss, {
            "loss_bc": self._value(loss.detach()),
            "grad_bc": self._grad_norm(loss, self.actor.param_actor()),
        }

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_normal, metrics_normal = self.loss_normal(batch)
        loss_bc, metrics_bc = self.loss_bc(batch)
        loss = self.weight_td3 * loss_normal + loss_bc
        return loss, {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        } | metrics_normal | metrics_bc
