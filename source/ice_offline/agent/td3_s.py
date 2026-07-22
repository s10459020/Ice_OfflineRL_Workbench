import torch

from ice_offline.agent.td3 import TD3Agent
from ice_offline.dataset._types import Batch


class TD3SAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_td3 = config.get("weight_td3", 0.01)
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_td3",
            "grad_td3",
            "loss_actor",
            "grad_actor",
            "param_q",
            "target_q",
        ]

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td3, metrics_td3 = self.loss_td3(batch)
        loss = self.weight_td3 * loss_td3
        return loss, {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        } | metrics_td3
