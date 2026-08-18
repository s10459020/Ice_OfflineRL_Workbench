from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.gp import GPAgent
from joint_learning.agents.scaspl import SCASPLAgent


class SCASPLGPAgent(GPAgent, SCASPLAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, lambda_s: float = 0.25, lambda_p: float = 2.5, actor_num_sample: int = 16, critic_rate_decay: float = 0.005, lambda_gp: float = 1.0, gp_count: int = 16, gp_threshold: float = 1.0, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, lambda_s=lambda_s, lambda_p=lambda_p, actor_num_sample=actor_num_sample, critic_rate_decay=critic_rate_decay, lambda_gp=lambda_gp, gp_count=gp_count, gp_threshold=gp_threshold, device=device)
