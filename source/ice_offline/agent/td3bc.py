from dataclasses import dataclass

import torch

from ice_offline.agent._spec import AgentBatch
from ice_offline.agent.td3 import TD3Agent


@dataclass
class TD3BCAgent(TD3Agent):
    alpha: float = 2.5

    # ====================
    # Actor loss
    # ====================
    def loss_bc(self, batch: AgentBatch) -> torch.Tensor:
        o, a, _, _, _ = batch
        a_pred = self.actor.pi(o)
        return ((a - a_pred) ** 2).mean()

    def loss_actor(self, batch: AgentBatch) -> torch.Tensor:
        return self.alpha * self.loss_td3(batch) + self.loss_bc(batch)
