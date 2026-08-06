from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scas_gp import SCASGPAgent
from joint_learning.datasets.lib import Batch


class SCASGPNAgent(SCASGPAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics: SCASDynamics, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamics=dynamics, device=device)
        self.weight_correction = 0.002

    def loss_actor(self, batch: Batch):
        # SCAS-GPN actor loss:
        #   L_normal = -E[Q(s, pi(s))] / E[|Q(s, pi(s))|]
        #   L_pi = (1 - lambda_s) * L_normal + lambda_s * L_SCAS.
        # The critic uses TD plus action-gradient penalty.
        return (1.0 - self.weight_correction) * self.loss_normal(batch) + self.weight_correction * self.loss_correction(batch)
