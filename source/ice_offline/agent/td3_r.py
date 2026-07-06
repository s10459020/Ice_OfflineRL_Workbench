import torch

from ice_offline.agent.td3 import TD3Agent
from ice_offline.dataset._types import Batch


class TD3RAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.weight_r = config.get("weight_r", 1e-4)

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_r",
            "grad_r",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "target_q",
        ]

    def loss_r(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = torch.zeros((), device=self.device)
        for param in self.critic.param_critic():
            loss = loss + param.square().sum()
        return loss, {
            "loss_r": self._value(loss.detach()),
            "grad_r": self._grad_norm(loss, self.critic.param_critic()),
        }

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td, metrics_td = self.loss_td(batch)
        loss_r, metrics_r = self.loss_r(batch)
        loss = loss_td + self.weight_r * loss_r
        return loss, metrics_td | metrics_r | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
