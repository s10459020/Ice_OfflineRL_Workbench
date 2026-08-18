from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.n import NAgent
from joint_learning.agents.scas import SCASAgent


class SCASNAgent(NAgent, SCASAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, lambda_s: float = 0.005, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, lambda_s=lambda_s, device=device)
