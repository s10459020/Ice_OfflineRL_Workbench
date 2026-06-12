from dataclasses import dataclass
from typing import ClassVar

import torch

from ice_offline.agent.cql_soft_q import CQLSoftQAgent


@dataclass
class CQLMaxQAgent(CQLSoftQAgent):
    agent_name: ClassVar[str] = "cql_max_q"
    # ====================
    # Critic target
    # ====================
    def target_sac(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # CQL max-Q backup:
        # y = r + gamma * max_a' min Q_target(s', a') * (1 - done)
        with torch.no_grad():
            an, _ = self.actor.sample_n(on)
            tq = self.critic.eval_tq_n(on, an).squeeze(-1)  # (Q, B, N)
            tq = tq.min(dim=0).values.max(dim=1, keepdim=True).values
            return r + self.gamma * tq * (1.0 - d)
