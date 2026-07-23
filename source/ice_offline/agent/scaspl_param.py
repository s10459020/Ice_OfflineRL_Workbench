import torch

from ice_offline.agent.scaspl import ScasplAgent
from ice_offline.dataset._types import Batch


class ScasplParamAgent(ScasplAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_pi = config.get("weight_pi", 0.005)
        self.weight_cor = config.get("weight_cor", 0.0)
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config=config,
            device=device,
        )

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "loss_correction",
            "grad_correction",
            "weight_pi",
            "weight_cor",
            "loss_actor",
            "grad_actor",
            "q_avg",
            "target_q",
        ]

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        loss_td3, metrics_td3 = self.loss_td3(batch)
        loss_correction, metrics_correction = self.loss_correction(batch)
        loss = self.weight_pi * loss_td3 + self.weight_cor * loss_correction
        return loss, metrics_td3 | metrics_correction | {
            "weight_pi": self.weight_pi,
            "weight_cor": self.weight_cor,
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        }
