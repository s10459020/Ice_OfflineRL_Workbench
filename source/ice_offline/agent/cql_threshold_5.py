from ice_offline.agent.cql import CQLAgent


class CQLThreshold5Agent(CQLAgent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            config={"threshold": 5.0} | config,
            device=device,
        )
