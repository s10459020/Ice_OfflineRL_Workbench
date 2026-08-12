from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scaspl import SCASPLAgent
from joint_learning.lib.dataset import Batch


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

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_policy(self, batch: Batch):
        # SCASPL-N actor loss:
        # Loss_Normalized_Policy = -E[Q_(min)(s,\pi(s))]/E[|Q_(min)(s,\pi(s))|]
        # Loss_Policy = (1 - \lambda_s) * Loss_Normalized_Policy + \lambda_s * Loss_State_Correction
        # The critic still uses SCASPL pseudo-label punishment.
        return (1.0 - self.weight_correction) * self.loss_normalized_policy(batch) + self.weight_correction * self.loss_state_correction(batch)
