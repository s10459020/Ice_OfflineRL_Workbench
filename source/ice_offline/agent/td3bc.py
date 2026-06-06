from dataclasses import dataclass
from typing import ClassVar

import torch

from ice_offline.agent.td3 import TD3Agent
from ice_offline.dataset._types import Batch


@dataclass
class TD3BCAgent(TD3Agent):
    agent_name: ClassVar[str] = "td3bc"
    alpha: float = 2.5

    # ====================
    # Actor loss
    # ====================
    def loss_bc(self, batch: Batch) -> torch.Tensor:
        o, a, _, _, _ = batch
        a_pred = self.actor.pi(o)
        return ((a - a_pred) ** 2).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        return self.alpha * self.loss_td3(batch) + self.loss_bc(batch)

