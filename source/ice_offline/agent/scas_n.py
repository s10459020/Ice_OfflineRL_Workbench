import torch

from ice_offline.agent.scas import ScasAgent
from ice_offline.dataset._types import Batch


class ScasNAgent(ScasAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config={"weight_correction": 0.001} | config,
            device=device,
        )

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_normal",
            "grad_normal",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
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
        loss_normal, metrics_normal = self.loss_normal(batch)
        loss_correction, metrics_correction = self.loss_correction(batch)
        loss = (
            (1.0 - self.weight_correction) * loss_normal
            + self.weight_correction * loss_correction
        )
        return loss, metrics_normal | metrics_correction | {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        }
