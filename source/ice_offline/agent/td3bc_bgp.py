from ice_offline.agent.td3bc_b import TD3BCBAgent
from ice_offline.agent.td3bc_gp import TD3BCGPAgent


class TD3BCBGPAgent(TD3BCBAgent, TD3BCGPAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_gp",
            "grad_gp",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "loss_bc",
            "grad_bc",
            "loss_actor",
            "grad_actor",
            "target_q",
            "grad_norm",
        ]
