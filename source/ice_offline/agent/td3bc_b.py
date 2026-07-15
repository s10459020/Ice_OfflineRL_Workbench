import torch

from ice_offline.agent.td3bc import TD3BCAgent
from ice_offline.dataset._types import Batch


class TD3BCBAgent(TD3BCAgent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_bc = config.get("weight_bc", 10000.0)
        super().__init__(obs_size=obs_size, act_size=act_size, config={"weight_td3": 100.0} | config, device=device)

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td3, metrics_td3 = self.loss_td3(batch)
        loss_bc, metrics_bc = self.loss_bc(batch)
        loss = self.weight_td3 * loss_td3 + self.weight_bc * loss_bc
        return loss, {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        } | metrics_td3 | metrics_bc
