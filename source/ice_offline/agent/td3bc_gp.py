from ice_offline.agent.td3_gp import TD3GPAgent
from ice_offline.agent.td3bc import TD3BCAgent


class TD3BCGPAgent(TD3BCAgent, TD3GPAgent):
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
