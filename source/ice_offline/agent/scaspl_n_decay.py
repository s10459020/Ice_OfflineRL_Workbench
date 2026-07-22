import torch

from ice_offline.agent.scaspl_n import ScasplNAgent
from ice_offline.dataset._types import Batch


class ScasplNDecayAgent(ScasplNAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, dynamics=dynamics, config=config, device=device)
        self.weight_punish_start = 0.5
        self.weight_punish_end = self.weight_punish_start / 100.0
        self.weight_punish_decay_steps = 100_000
        self.weight_punish = self.weight_punish_start

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "weight_punish",
            "loss_critic",
            "grad_critic",
            "loss_normal",
            "grad_normal",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "q_avg",
            "target_q",
        ]

    def current_weight_punish(self) -> float:
        progress = min(float(self.update_step) / float(self.weight_punish_decay_steps), 1.0)
        return self.weight_punish_start + (self.weight_punish_end - self.weight_punish_start) * progress

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        loss_td, metrics_td = self.loss_td(batch)
        loss_punish, metrics_punish = self.loss_punish(batch)
        weight_punish = self.current_weight_punish()
        self.weight_punish = weight_punish
        loss = loss_td + weight_punish * loss_punish
        return loss, metrics_td | metrics_punish | {
            "weight_punish": weight_punish,
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
