import torch

from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.td3_r import TD3RAgent
from ice_offline.dataset._types import Batch


class AsplRAgent(AsplAgent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.weight_r = config.get("weight_r", 1)

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_r",
            "grad_r",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "q_avg",
            "target_q",
        ]

    def loss_r(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return TD3RAgent.loss_r(self, batch)

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td, metrics_td = self.loss_td(batch)
        loss_punish, metrics_punish = self.loss_punish(batch)
        loss_r, metrics_r = self.loss_r(batch)
        loss = loss_td + self.weight_punish * loss_punish + self.weight_r * loss_r
        return loss, metrics_td | metrics_punish | metrics_r | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
