from dataclasses import dataclass

import torch

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.td3 import TD3Agent
from ice_offline.dataset._types import Batch


@dataclass
class TD3BCAgent(TD3Agent):
    id: str = "td3bc"
    alpha: float = 2.5

    # ====================
    # Update
    # ====================
    def update_with_metrics(self, batch: Batch) -> MetricValues:
        self.update_step += 1
        loss_critic = self.loss_critic(batch)
        grad_critic = self._grad_norm(loss_critic, self.critic.parameters())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        metrics = {
            "loss_critic": loss_critic.detach(),
            "grad_critic": grad_critic.detach(),
            "loss_td3": None,
            "grad_td3": None,
            "loss_bc": None,
            "grad_bc": None,
            "loss_actor": None,
            "grad_actor": None,
        }

        if self.update_step % self.update_actor_interval == 0:
            loss_td3 = self.loss_td3(batch)
            loss_bc = self.loss_bc(batch)
            loss_actor = self.alpha * loss_td3 + loss_bc
            grad_td3 = self._grad_norm(loss_td3, self.actor.parameters())
            grad_bc = self._grad_norm(loss_bc, self.actor.parameters())
            grad_actor = self._grad_norm(loss_actor, self.actor.parameters())

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            self.critic.update_target_soft()
            self.actor.update_target_soft()

            metrics.update({
                "loss_td3": loss_td3.detach(),
                "grad_td3": grad_td3.detach(),
                "loss_bc": loss_bc.detach(),
                "grad_bc": grad_bc.detach(),
                "loss_actor": loss_actor.detach(),
                "grad_actor": grad_actor.detach(),
            })

        return metrics

    # ====================
    # Actor loss
    # ====================
    def loss_bc(self, batch: Batch) -> torch.Tensor:
        o, a, _, _, _ = batch
        a_pred = self.actor.pi(o)
        return ((a - a_pred) ** 2).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        return self.alpha * self.loss_td3(batch) + self.loss_bc(batch)

