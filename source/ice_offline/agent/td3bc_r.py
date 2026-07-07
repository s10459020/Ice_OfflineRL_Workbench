from ice_offline.agent.td3_r import TD3RAgent
from ice_offline.agent.td3bc import TD3BCAgent


class TD3BCRAgent(TD3BCAgent, TD3RAgent):
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_r",
            "grad_r",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "loss_bc",
            "grad_bc",
            "loss_actor",
            "grad_actor",
            "target_q",
        ]
