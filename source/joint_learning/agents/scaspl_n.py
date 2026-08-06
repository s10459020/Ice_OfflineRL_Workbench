from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scaspl import SCASPLAgent
from joint_learning.datasets.lib import Batch


class SCASPLNAgent(SCASPLAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics: SCASDynamics, device: str = "cuda") -> None:
        super().__init__(
            obs_size,
            act_size,
            dynamics=dynamics,
            weight_correction=0.001,
            weight_punish=0.005,
            device=device,
        )

    def loss_actor(self, batch: Batch):
        # SCASPL-N actor loss:
        #   L_normal = -E[Q(s, pi(s))] / E[|Q(s, pi(s))|]
        #   L_pi = (1 - lambda_s) * L_normal + lambda_s * L_SCAS.
        # The critic still uses SCASPL pseudo-label punishment.
        return (1.0 - self.weight_correction) * self.loss_normal(batch) + self.weight_correction * self.loss_correction(batch)
