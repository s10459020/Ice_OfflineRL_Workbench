from dataclasses import dataclass
from typing import ClassVar

import torch

from ice_offline.agent.scas_min import ScasDynamic
from ice_offline.agent.scas_min import ScasMinAgent


@dataclass
class ScasMeanAgent(ScasMinAgent):
    agent_name: ClassVar[str] = "scas_mean"
    # ====================
    # Critic loss
    # ====================
    def target_td3(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # use tq_mean instead of tq_min to compute target value
        with torch.no_grad():
            an = self.actor.noise_action(self.actor.tpi(on))
            tq = self.critic.tq_mean(on, an)
            return r + self.gamma * tq * (1 - d)


