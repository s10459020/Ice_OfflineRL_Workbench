import torch

from ice_offline.agent.scas import ScasAgent
from ice_offline.dataset._types import Batch


class ScasNAgent(ScasAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_normal",
            "grad_normal",
            "target_q",
        ]

    def loss_normal(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, _, _, _, _ = batch
        a = self.actor.pi(o)
        q = self.critic.q_min(o, a)
        loss = -q.mean() / q.abs().mean().detach()
        return loss, {
            "loss_normal": self._value(loss.detach()),
            "grad_normal": self._grad_norm(loss, self.actor.param_actor()),
        }

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss_normal(batch)
