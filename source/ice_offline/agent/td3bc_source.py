from dataclasses import dataclass

import torch

from ice_offline.agent._spec import AgentBatch
from ice_offline.agent.td3 import TD3Agent


@dataclass
class TD3BCSourceAgent(TD3Agent):
    alpha: float = 2.5

    # ====================
    # Actor mathmatics
    # ====================
    def loss_bc(self, batch: AgentBatch, a_pred: torch.Tensor) -> torch.Tensor:
        # loss = E{s,a~D}[ MSE(a - pi(s)) ]
        _, a, _, _, _ = batch
        return ((a - a_pred) ** 2).mean()

    def loss_td3(self, batch: AgentBatch, a_pred: torch.Tensor) -> torch.Tensor:
        # use only q1 for actor update
        o, _, _, _, _ = batch
        q = self.critic.q_networks[0](o, a_pred)
        lam = self.alpha / q.abs().mean().detach()
        return lam * -q.mean()

    def loss_actor(self, batch: AgentBatch) -> torch.Tensor:
        o, _, _, _, _ = batch
        a_pred = self.actor.pi(o)
        return self.loss_td3(batch, a_pred) + self.loss_bc(batch, a_pred)
