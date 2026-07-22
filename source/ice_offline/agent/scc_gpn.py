from ice_offline.agent.scc_gp import SccGPAgent
from ice_offline.agent.scc_n import SccNAgent


class SccGPNAgent(SccNAgent, SccGPAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_conservative",
            "grad_conservative",
            "loss_gp",
            "grad_gp",
            "loss_critic",
            "grad_critic",
            "loss_normal",
            "grad_normal",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "target_q",
            "grad_norm",
        ]
