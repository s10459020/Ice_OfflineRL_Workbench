import torch

from ice_offline.agent.scaspl_n import ScasplNAgent
from ice_offline.dataset._types import Batch


class ScasplPNAgent(ScasplNAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        config = {"lambda_td": 0.5, "weight_pi": 1.0, "weight_correction": 0.001, "weight_punish": 0.5} | config
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config=config,
            device=device,
        )
        self.lambda_td = config.get("lambda_td", 0.5)
        self.weight_pi = config.get("weight_pi", 1.0)

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td, metrics_td = self.loss_td(batch)
        loss_punish, metrics_punish = self.loss_punish(batch)
        loss = self.lambda_td * loss_td + self.weight_punish * loss_punish
        return loss, metrics_td | metrics_punish | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_normal, metrics_normal = self.loss_normal(batch)
        loss_correction, metrics_correction = self.loss_correction(batch)
        loss = self.weight_pi * loss_normal + self.weight_correction * loss_correction
        return loss, metrics_normal | metrics_correction | {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        }
