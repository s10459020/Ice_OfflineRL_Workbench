from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scc_gp import SCCGPAgent
from joint_learning.lib.dataset import Batch


class SCCGPNAgent(SCCGPAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics: SCASDynamics, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamics=dynamics, device=device)
        self.weight_correction = 0.01

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_policy(self, batch: Batch):
        # SCC-GPN actor loss:
        # Loss_Normalized_Policy = -E[Q_(min)(s,\pi(s))]/E[|Q_(min)(s,\pi(s))|]
        # Loss_Policy = (1 - \lambda_s) * Loss_Normalized_Policy + \lambda_s * Loss_State_Correction
        # The critic uses TD, SCC conservative margin, and action-gradient penalty.
        return (1.0 - self.weight_correction) * self.loss_normalized_policy(batch) + self.weight_correction * self.loss_state_correction(batch)
