from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.gp import GPAgent
from joint_learning.agents.scas_n import SCASNAgent


class SCASGPNAgent(GPAgent, SCASNAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, lambda_s: float = 0.002, lambda_gp: float = 1.0, gp_count: int = 16, gp_threshold: float = 1.0, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, lambda_s=lambda_s, lambda_gp=lambda_gp, gp_count=gp_count, gp_threshold=gp_threshold, device=device)
