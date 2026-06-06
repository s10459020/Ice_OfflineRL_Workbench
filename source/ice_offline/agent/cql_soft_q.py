from dataclasses import dataclass
from typing import ClassVar

import math
import torch

from ice_offline.dataset._types import Batch
from ice_offline.agent.sac import SACAgent
from ice_offline.agent.sac import SACActor
from ice_offline.agent.sac import SACCritic


class _CQLActor(SACActor):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        n_action_samples: int = 10,
    ):
        super().__init__(obs_size, act_size)
        self.act_size = act_size
        self.n_action_samples = n_action_samples

    def sample_random_n(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = o.shape[0]
        a = torch.zeros(
            (batch_size * self.n_action_samples, self.act_size),
            device=o.device,
        ).uniform_(-1.0, 1.0)
        log_prob = torch.full(
            (batch_size * self.n_action_samples, 1),
            math.log(0.5**self.act_size),
            device=o.device,
        )
        return a, log_prob


class _CQLConservativeMultiplier(torch.nn.Module):
    def __init__(self, learning_rate: float = 1e-4, initial_multiplier: float = 1.0):
        super().__init__()
        self.log_cql_multiplier = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))
        self.log_cql_multiplier.data.fill_(math.log(initial_multiplier))
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    def forward(self) -> torch.Tensor:
        return self.log_cql_multiplier.exp().clamp(0.0, 1e6)

    def update(self, conservative_loss_detached: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        multiplier_loss = self.loss(conservative_loss_detached)
        multiplier_loss.backward()
        self.optimizer.step()

    def loss(self, conservative_loss_detached: torch.Tensor) -> torch.Tensor:
        # loss = -E[ alpha * L_cql ]
        # Lagrangian dual，若L_cql項長期偏大，則加強修改力度
        return -(self() * conservative_loss_detached).mean()


class _CQLCritic(SACCritic):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        alpha_threshold: float = 10.0,
        conservative_weight: float = 5.0,
        n_action_samples: int = 10,
    ):
        super().__init__(obs_size, act_size)
        self.alpha_threshold = alpha_threshold
        self.conservative_weight = conservative_weight
        self.n_action_samples = n_action_samples

    def eval_q_n(self, o: torch.Tensor, a_sample: torch.Tensor) -> torch.Tensor:
        o = o.repeat_interleave(self.n_action_samples, dim=0)  # (B, O) => (B*N, O)
        a = a_sample.reshape(-1, a_sample.shape[-1])           # (B*N, A)
        return torch.stack([q(o, a) for q in self.q_networks], dim=0)

    def eval_tq_n(self, o: torch.Tensor, a_sample: torch.Tensor) -> torch.Tensor:
        o = o.repeat_interleave(self.n_action_samples, dim=0)
        a = a_sample.reshape(-1, a_sample.shape[-1])
        return torch.stack([tq(o, a) for tq in self.tq_networks], dim=0)


@dataclass
class CQLSoftQAgent(SACAgent):
    agent_name: ClassVar[str] = "cql_soft_q"
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 1e-4
    cql_multiplier_learning_rate: float = 1e-4

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        super().__post_init__()
        self.actor = _CQLActor(
            obs_size=self.obs_size,
            act_size=self.act_size,
        ).to(self.device)
        self.critic = _CQLCritic(
            obs_size=self.obs_size,
            act_size=self.act_size,
        ).to(self.device)
        self.multiplier = _CQLConservativeMultiplier(
            learning_rate=self.cql_multiplier_learning_rate,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(),
            lr=self.actor_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.q_networks.parameters(),
            lr=self.critic_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, object]:
        state = super()._save_dict()
        state["multiplier"] = self.multiplier.state_dict()
        state["multiplier_optimizer"] = self.multiplier.optimizer.state_dict()
        return state

    def _load_dict(self, state: dict[str, object]) -> None:
        super()._load_dict(state)
        self.multiplier.load_state_dict(state["multiplier"])
        self.multiplier.optimizer.load_state_dict(state["multiplier_optimizer"])

    # ====================
    # Critic loss
    # ====================
    def loss_suppress(self, batch: Batch) -> torch.Tensor:
        # L_CQL(H) = E_D[s]{log *              sum_a[exp(Q)] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         sum_a[p* exp(Q)/p] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         E_a[exp(Q-log(p))] } - E_D[s,a]{Q}
        #         ~= E_D[s]{log * 1/N * sum_N[exp(Q-log(p))] } - E_D[s,a]{Q} # sample approximation
        #         => logsumexp(Q-log(p)) - E_(s,a)~D[Q]  # 單步loss     
        #
        # E_D[s]: input o
        # E_D[s,a]: input o,a
        # CQL sample approximation: a ~ p(a) => Uniform/ pi(.|s)/ pi(.|s') 三種N次
        o, a, _, on, _ = batch
        batch_size = o.shape[0]

        a_s, logp = self.actor.sample_n(o, self.critic.n_action_samples)
        an, logpn = self.actor.sample_n(on, self.critic.n_action_samples)
        ar, logpr = self.actor.sample_random_n(o)

        q = self.critic.eval_q_n(o, a_s).view(2, batch_size, self.critic.n_action_samples)
        qn = self.critic.eval_q_n(o, an).view(2, batch_size, self.critic.n_action_samples)
        qr = self.critic.eval_q_n(o, ar).view(2, batch_size, self.critic.n_action_samples)
        q_cat = torch.cat([q, qn, qr], dim=2)               # (2,B,3N)

        logp = logp.view(1, batch_size, self.critic.n_action_samples)   # (1,B,N)
        logpn = logpn.view(1, batch_size, self.critic.n_action_samples) # (1,B,N)
        logpr = logpr.view(1, batch_size, self.critic.n_action_samples) # (1,B,N)
        logp_cat = torch.cat([logp, logpn, logpr], dim=2)   # (1,B,3N)

        logsumexp = torch.logsumexp(q_cat - logp_cat, dim=2, keepdim=True)  # (2,B,1)
        data_q = torch.stack(self.critic.q_all(o, a), dim=0)
        return (logsumexp - data_q).mean(dim=[1, 2])        # (2), double Q

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # CQL loss: loss_td + multiplier * loss_suppress
        loss_td = self.loss_td(batch)
        loss_suppress = self.loss_suppress(batch)            # (2,)
        loss_suppress = self.critic.conservative_weight * (
            loss_suppress - self.critic.alpha_threshold
        )
        if torch.is_grad_enabled():
            self.multiplier.update(loss_suppress.detach())
        return loss_td + (self.multiplier() * loss_suppress).sum()

