from joint_learning.agents.td3bc_xn import TD3BCXNAgent


class TD3BCPAgent(TD3BCXNAgent):
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, lambda_td3=0.01, device=device)
