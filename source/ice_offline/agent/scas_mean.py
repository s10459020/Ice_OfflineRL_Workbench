from dataclasses import dataclass

import torch

from ice_offline.agent.scas_min import ScasDynamicAgent
from ice_offline.agent.scas_min import ScasMinAgent


@dataclass
class ScasMeanAgent(ScasMinAgent):
    # ====================
    # Critic loss
    # ====================
    def target_td3(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # target = r + gamma * Q(o',a') * (1-done)
        # TD3: noise action a' for policy smoothing
        with torch.no_grad():
            an = self.actor.noise_action(self.actor.tpi(on))
            tq = self.critic.tq_mean(on, an)
            return r + self.gamma * tq * (1 - d)
