from ice_offline.agent.td3_gp import TD3GPAgent
from ice_offline.agent.td3_n import TD3NAgent


class TD3GPNAgent(TD3NAgent, TD3GPAgent):
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
            "target_q",
            "grad_norm",
        ]
