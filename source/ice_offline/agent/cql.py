from dataclasses import dataclass
from typing import ClassVar

import torch

from ice_offline.agent.cql_soft_q import CQLSoftQAgent


@dataclass
class CQLAgent(CQLSoftQAgent):
    agent_name: ClassVar[str] = "cql"

    # ====================
    # Critic target
    # ====================
    def target_td(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # CQL deterministic backup:
        # y = r + gamma * min Q_target(s', pi_mode(s')) * (1-done)
        with torch.no_grad():
            an = self.actor(on)
            tq = self.critic.tq_min(on, an)
            return r + self.gamma * tq * (1.0 - d)

