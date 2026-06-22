from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.aspl import AsplActor
from ice_offline.agent.aspl import AsplCritic
from ice_offline.agent.scas import ScasAgent
from ice_offline.dataset._types import Batch


@dataclass
class ScasAsplAgent(ScasAgent, AsplAgent):
    aspl_alpha: float = 2.5

    def __post_init__(self) -> None:
        AsplAgent.__post_init__(self)
        self.dynamics = self.dynamics.prepare()

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch) -> None:
        _, _, r, sn, d = batch
        target = self.target_td3(sn, r, d)
        self.update_step += 1

        self.critic.update_moving_avg(target)

        self.critic_optimizer.zero_grad()
        loss_critic = self.loss_critic(batch, target)
        loss_critic.backward()
        self.critic_optimizer.step()

        if self.update_step % self.update_actor_interval == 0:
            self.actor_optimizer.zero_grad()
            loss_actor = self.loss_actor(batch)
            loss_actor.backward()
            self.actor_optimizer.step()

            self.critic.update_target_soft()
            self.actor.update_target_soft()

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        _, _, r, sn, d = batch
        target = self.target_td3(sn, r, d)
        self.update_step += 1

        moving_avg = self.critic.update_moving_avg(target)

        loss_td = self.loss_td(batch)
        grad_td = self._grad_norm(loss_td, self.critic.parameters())

        loss_punish = AsplAgent.loss_punish(self, batch)
        grad_punish = self._grad_norm(loss_punish, self.critic.parameters())

        loss_critic = loss_td + self.aspl_alpha * loss_punish
        grad_critic = self._grad_norm(loss_critic, self.critic.parameters())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        metrics = {
            "loss_td": loss_td.detach(),
            "grad_td": grad_td.detach(),
            "loss_punish": loss_punish.detach(),
            "grad_punish": grad_punish.detach(),
            "loss_critic": loss_critic.detach(),
            "grad_critic": grad_critic.detach(),
            "loss_actor": None,
            "grad_actor": None,
            "moving_avg": moving_avg.detach(),
            "target_q": target.abs().mean(),
        }

        if self.update_step % self.update_actor_interval == 0:
            loss_actor = self.loss_actor(batch)
            grad_actor = self._grad_norm(loss_actor, self.actor.parameters())

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            self.critic.update_target_soft()
            self.actor.update_target_soft()

            metrics.update({
                "loss_actor": loss_actor.detach(),
                "grad_actor": grad_actor.detach(),
            })

        return metrics

    # ====================
    # Critic loss
    # ====================
    def loss_critic(
        self,
        batch: Batch,
        q_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_target is None:
            _, _, r, sn, d = batch
            q_target = self.target_td3(sn, r, d)
        s, a, _, _, _ = batch
        loss_td = sum(F.mse_loss(q, q_target) for q in self.critic.q_all(s, a))
        loss_punish = AsplAgent.loss_punish(self, batch)
        return loss_td + self.aspl_alpha * loss_punish

