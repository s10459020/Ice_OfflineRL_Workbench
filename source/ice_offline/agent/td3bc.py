import torch

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.td3 import TD3Agent
from ice_offline.dataset._types import Batch


class TD3BCAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_td3 = config.get("weight_td3", 2.5)
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)

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
            "loss_normal": None,
            "grad_normal": None,
            "loss_bc": None,
            "grad_bc": None,
            "loss_actor": None,
            "grad_actor": None,
        }

        if self.update_step % self.update_actor_interval == 0:
            loss_normal = self.loss_normal(batch)
            loss_bc = self.loss_bc(batch)
            loss_actor = self.weight_td3 * loss_normal + loss_bc
            grad_normal = self._grad_norm(loss_normal, self.actor.parameters())
            grad_bc = self._grad_norm(loss_bc, self.actor.parameters())
            grad_actor = self._grad_norm(loss_actor, self.actor.parameters())

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            self.critic.update_target_soft()
            self.actor.update_target_soft()

            metrics.update({
                "loss_normal": loss_normal.detach(),
                "grad_normal": grad_normal.detach(),
                "loss_bc": loss_bc.detach(),
                "grad_bc": grad_bc.detach(),
                "loss_actor": loss_actor.detach(),
                "grad_actor": grad_actor.detach(),
            })

        return metrics

    # ====================
    # Actor loss
    # ====================
    def loss_normal(self, batch: Batch) -> torch.Tensor:
        o, _, _, _, _ = batch
        a = self.actor.pi(o)
        q = self.critic.q_min(o, a)
        return - q.mean() / q.abs().mean().detach()

    def loss_bc(self, batch: Batch) -> torch.Tensor:
        o, a, _, _, _ = batch
        a_pred = self.actor.pi(o)
        return ((a - a_pred) ** 2).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        return self.weight_td3 * self.loss_normal(batch) + self.loss_bc(batch)

