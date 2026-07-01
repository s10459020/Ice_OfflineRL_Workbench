import torch

from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.scas_gp import ScasGPAgent
from ice_offline.agent.scaspl import ScasplAgent
from ice_offline.dataset._types import Batch


class ScasplGPAgent(ScasplAgent, ScasGPAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, dynamics=dynamics, config=config, device=device)
        self.weight_gp = config.get("weight_gp", 0.1)
        self.gp_threshold = config.get("gp_threshold", 100.0)
        self.gp_interval = config.get("gp_interval", 5)
        self.gp_count = config.get("gp_count", 16)

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_gp",
            "grad_gp",
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
            "grad_norm",
        ]

    def update(self, batch: Batch) -> dict[str, torch.Tensor]:
        _, _, r, sn, d = batch
        self.update_step += 1

        if self.update_step % self.gp_interval != 0:
            metrics = self.update_aspl(batch)
        else:
            metrics = self.update_critic(batch)
            
        target = self.target_td3(sn, r, d)
        q_avg = self.critic.update_q_avg(target)
        metrics["q_avg"] = self._value(q_avg.detach())

        if self.update_step % self.update_actor_interval == 0:
            metrics |= self.update_actor(batch)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

        return metrics

    def update_aspl(self, batch: Batch) -> dict[str, torch.Tensor]:
        loss_critic, metrics = AsplAgent.loss_critic(self, batch)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        return metrics

    def update_critic(self, batch: Batch) -> dict[str, torch.Tensor]:
        loss_critic, metrics = self.loss_critic(batch)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        return metrics

    # ====================
    # Critic loss
    # ====================
    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_scaspl, metrics_scaspl = AsplAgent.loss_critic(self, batch)
        loss_gp, metrics_gp = self.loss_gp(batch)
        loss = loss_scaspl + self.weight_gp * loss_gp
        return loss, metrics_scaspl | metrics_gp | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
