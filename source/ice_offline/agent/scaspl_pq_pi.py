from ice_offline.agent.scaspl_pq_base import ScasplPQBaseAgent


class ScasplPQPiAgent(ScasplPQBaseAgent):
    normal_q_source = "punish"
    correction_q_source = "value"

    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        config = {
            "weight_pi": 0.01,
            "weight_correction": 0.01,
            "weight_punish": 2.5,
        } | config
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config=config,
            device=device,
        )
