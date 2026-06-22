from dataclasses import dataclass

import torch

from ice_offline.agent.scas_pre import ScasDynamic
from ice_offline.agent.scas_pre import ScasPreAgent
from ice_offline.agent.td3 import TD3Actor
from ice_offline.agent.td3 import TD3Critic
from ice_offline.dataset._types import Batch


@dataclass
class ScasAgent(ScasPreAgent):
    def __post_init__(self) -> None:
        self.dynamics = ScasDynamic(
            obs_size=self.obs_size,
            act_size=self.act_size,
            device=self.device,
        )
        self.actor = TD3Actor(self.obs_size, self.act_size).to(self.device)
        self.critic = TD3Critic(self.obs_size, self.act_size, q_count=self.q_count).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.pi.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.q_networks.parameters(),
            lr=self.critic_learning_rate,
        )

    def update(self, batch: Batch):
        self.dynamics.update(batch)
        super().update(batch)

    def _save_dict(self) -> dict[str, object]:
        state = super()._save_dict()
        state["dynamics"] = self.dynamics._save_dict()
        return state

    def _load_dict(self, state: dict[str, object]) -> None:
        super()._load_dict(state)
        self.dynamics._load_dict(state["dynamics"])
