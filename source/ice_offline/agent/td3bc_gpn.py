from ice_offline.agent.td3_gp import TD3GPAgent
from ice_offline.agent.td3bc_n import TD3BCNAgent


class TD3BCGPNAgent(TD3BCNAgent, TD3GPAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_gp",
            "grad_gp",
            "loss_critic",
            "grad_critic",
            "loss_normal",
            "grad_normal",
            "loss_bc",
            "grad_bc",
            "loss_actor",
            "grad_actor",
            "target_q",
            "grad_norm",
        ]
