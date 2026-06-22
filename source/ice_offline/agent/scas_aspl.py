import torch
import torch.nn.functional as F

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.aspl import AsplActor
from ice_offline.agent.aspl import AsplCritic
from ice_offline.agent.scas import ScasAgent
from ice_offline.dataset._types import Batch


class ScasAsplAgent(ScasAgent, AsplAgent):
    weight_punish: float = 2.5

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamics,
        config: dict[str, object] = {},
        device: str = "cuda",
    ) -> None:
        cfg = config
        self.weight_punish = cfg.get("weight_punish", 2.5)
        self.id = str(cfg.get("id", "aspl"))
        self.learning_rate = cfg.get("learning_rate", 3e-4)
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config=cfg,
            device=device,
        )
        self.actor = AsplActor(
            obs_size=self.obs_size,
            act_size=self.act_size,
            seed=cfg.get("actor_seed", 42),
            num_sample=cfg.get("actor_num_sample", 5),
        ).to(self.device)
        self.critic = AsplCritic(
            obs_size=self.obs_size,
            act_size=self.act_size,
            q_count=self.q_count,
            rate_decay=cfg.get("critic_rate_decay", 0.005),
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.pi.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.q_networks.parameters(), lr=self.learning_rate)
        self.dynamics = self.dynamics.prepare()

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch) -> None:
        _, _, r, sn, d = batch
        target = self.target_td3(sn, r, d)
        self.update_step += 1

        self.critic.update_scale_punish(target)

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

        scale_punish = self.critic.update_scale_punish(target)

        loss_td = self.loss_td(batch)
        grad_td = self._grad_norm(loss_td, self.critic.parameters())

        loss_punish = AsplAgent.loss_punish(self, batch)
        grad_punish = self._grad_norm(loss_punish, self.critic.parameters())

        loss_critic = loss_td + self.weight_punish * loss_punish
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
            "scale_punish": scale_punish.detach(),
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
        return loss_td + self.weight_punish * loss_punish

