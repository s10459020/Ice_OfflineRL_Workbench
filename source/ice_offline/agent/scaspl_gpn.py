from ice_offline.agent.scas_gp import ScasGPAgent
from ice_offline.agent.scaspl_n import ScasplNAgent


class ScasplGPNAgent(ScasplNAgent, ScasGPAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_gp",
            "grad_gp",
            "loss_critic",
            "grad_critic",
            "loss_normal",
            "grad_normal",
            "q_avg",
            "target_q",
            "grad_norm",
        ]
