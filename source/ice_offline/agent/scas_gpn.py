from ice_offline.agent.scas_gp import ScasGPAgent
from ice_offline.agent.scas_n import ScasNAgent


class ScasGPNAgent(ScasNAgent, ScasGPAgent):
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
