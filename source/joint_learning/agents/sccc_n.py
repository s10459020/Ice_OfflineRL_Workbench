from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.n import NAgent
from joint_learning.agents.sccc import SCCCAgent


class SCCCNAgent(NAgent, SCCCAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, lambda_s: float = 0.005, lambda_q: float = 10.0, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, lambda_s=lambda_s, lambda_q=lambda_q, device=device)
