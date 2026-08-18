from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.n import NAgent
from joint_learning.agents.scaspl import SCASPLAgent


class SCASPLNAgent(NAgent, SCASPLAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, lambda_s: float = 0.005, lambda_p: float = 0.5, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, lambda_s=lambda_s, lambda_p=lambda_p, device=device)
