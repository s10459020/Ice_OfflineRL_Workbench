from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scc import SCCAgent
from joint_learning.lib.dataset import Batch


class SCCNAgent(SCCAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics: SCASDynamics, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamics=dynamics, weight_conservative=10.0, device=device)
        self.weight_correction = 0.01

    def loss_actor(self, batch: Batch):
        # SCC-N actor loss:
        #   L_normal = -E[Q(s, pi(s))] / E[|Q(s, pi(s))|]
        #   L_pi = (1 - lambda_s) * L_normal + lambda_s * L_SCAS.
        # The critic still uses the SCC conservative margin loss.
        return (1.0 - self.weight_correction) * self.loss_normal(batch) + self.weight_correction * self.loss_correction(batch)
