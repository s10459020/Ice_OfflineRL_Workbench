import torch

from ice_offline.agent.cql import CQLAgent
from ice_offline.agent.sdc import SDCAgent
from ice_offline.dataset._types import Batch


class SDCCQLAgent(SDCAgent, CQLAgent):
    def __init__(self, obs_size: int, act_size: int, model, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, model=model, config=config, device=device)

    def update_critic(self, batch: Batch) -> None:
        loss_suppress = CQLAgent.loss_suppress(self, batch).detach()
        self.multiplier.optimizer.zero_grad()
        loss_multiplier = self.multiplier.loss(loss_suppress)
        loss_multiplier.backward()
        self.multiplier.optimizer.step()

        self.critic_optimizer.zero_grad()
        loss_critic = self.loss_critic(batch)
        loss_critic.backward()
        self.critic_optimizer.step()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        loss_td = self.loss_td(batch)
        loss_suppress = CQLAgent.loss_suppress(self, batch)
        return loss_td + self.multiplier() * (
            loss_suppress - self.multiplier.threshold
        )

