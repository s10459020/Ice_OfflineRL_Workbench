import torch

from ice_offline.agent.aspl_c import AsplCAgent
from ice_offline.agent.scaspl_gp import ScasplGPAgent
from ice_offline.dataset._types import Batch


class ScasplGPCAgent(ScasplGPAgent, AsplCAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_compensate",
            "grad_compensate",
            "loss_gp",
            "grad_gp",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "q_avg",
            "target_q",
            "grad_norm",
        ]

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_scaspl, metrics_scaspl = AsplCAgent.loss_critic(self, batch)
        loss_gp, metrics_gp = self.loss_gp(batch)
        loss = loss_scaspl + self.weight_gp * loss_gp
        return loss, metrics_scaspl | metrics_gp | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
