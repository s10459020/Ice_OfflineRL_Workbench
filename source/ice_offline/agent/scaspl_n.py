import torch

from ice_offline.agent.scaspl import ScasplAgent
from ice_offline.dataset._types import Batch


class ScasplNAgent(ScasplAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_critic",
            "grad_critic",
            "loss_normal",
            "grad_normal",
            "q_avg",
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
