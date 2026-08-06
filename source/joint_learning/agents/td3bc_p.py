from joint_learning.agents.td3bc_xn import TD3BCXNAgent


class TD3BCPAgent(TD3BCXNAgent):
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        # TD3BC-P keeps the non-normalized TD3BC-XN objective but lowers the TD3 weight.
        # This matches the paper's parameter-adjusted variant for restoring the actor loss ratio.
        super().__init__(obs_size, act_size, weight_td3=0.01, device=device)
