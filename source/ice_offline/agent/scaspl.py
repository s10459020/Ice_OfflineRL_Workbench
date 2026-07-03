import torch

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.aspl import AsplActor
from ice_offline.agent.aspl import AsplCritic
from ice_offline.agent.scas import ScasAgent
from ice_offline.dataset._types import Batch


class ScasplAgent(ScasAgent, AsplAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_punish = config.get("weight_punish", 2.5)
        super().__init__(obs_size=obs_size, act_size=act_size, dynamics=dynamics, config=config, device=device)
        self.actor = AsplActor(self.obs_size, self.act_size, config).to(self.device)
        self.critic = AsplCritic(self.obs_size, self.act_size, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.param_actor())
        self.critic_optimizer = torch.optim.Adam(self.critic.param_critic())
        self.dynamics = self.dynamics.prepare()

    # ====================
    # Update
    # ====================
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
            "loss_actor",
            "grad_actor",
            "q_avg",
            "target_q",
        ]

    def update(self, batch: Batch) -> MetricValues:
        _, _, r, sn, d = batch
        target = self.target_td3(sn, r, d)

        q_avg = self.critic.update_q_avg(target)
        metrics = self.update_critic(batch)
        metrics["q_avg"] = self._value(q_avg.detach())

        self.update_step += 1
        if self.update_step % self.update_actor_interval == 0:
            metrics |= self.update_actor(batch)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

        return metrics
