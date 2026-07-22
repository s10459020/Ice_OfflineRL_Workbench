from ice_offline.agent.aspl_c import AsplCAgent
from ice_offline.agent.scaspl import ScasplAgent


class ScasplCAgent(ScasplAgent, AsplCAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config=config,
            device=device,
        )

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_compensate",
            "grad_compensate",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "q_avg",
            "target_q",
        ]
