import torch

from ice_offline.agent.scas import ScasAgent
from ice_offline.dataset._types import Batch


class ScasAdjectAgent(ScasAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.lambda_td3 = config.get("lambda_td3", 0.01)
        self.lambda_corr = config.get("lambda_corr", 1.0)
        super().__init__(obs_size=obs_size, act_size=act_size, dynamics=dynamics, config=config, device=device)

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td3, metrics_td3 = self.loss_td3(batch)
        loss_correction, metrics_correction = self.loss_correction(batch)
        loss = self.lambda_td3 * loss_td3 + self.lambda_corr * loss_correction
        return loss, metrics_td3 | metrics_correction | {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        }
