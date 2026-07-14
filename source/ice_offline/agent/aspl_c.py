import torch

from ice_offline.agent.aspl import AsplAgent
from ice_offline.dataset._types import Batch


class AsplCAgent(AsplAgent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.weight_compensate = config.get("weight_compensate", 0.005)

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_compensate",
            "grad_compensate",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "q_avg",
            "target_q",
        ]

    def loss_compensate(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, a, _, _, _ = batch
        data_q = torch.stack(self.critic.q_all(o, a), dim=0)
        loss = -data_q.mean()
        return loss, {
            "loss_compensate": self._value(loss.detach()),
            "grad_compensate": self._grad_norm(loss, self.critic.param_critic()),
        }

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td, metrics_td = self.loss_td(batch)
        loss_punish, metrics_punish = self.loss_punish(batch)
        loss_compensate, metrics_compensate = self.loss_compensate(batch)
        loss = (
            loss_td
            + self.weight_punish * loss_punish
            + self.weight_compensate * loss_compensate
        )
        return loss, metrics_td | metrics_punish | metrics_compensate | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
