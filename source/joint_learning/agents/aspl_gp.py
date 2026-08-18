from joint_learning.agents.aspl import ASPLAgent
from joint_learning.agents.gp import GPAgent


class ASPLGPAgent(GPAgent, ASPLAgent):
    def __init__(self, obs_size: int, act_size: int, lambda_gp: float = 1.0, gp_count: int = 16, gp_threshold: float = 1.0, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, lambda_gp=lambda_gp, gp_count=gp_count, gp_threshold=gp_threshold, device=device)
