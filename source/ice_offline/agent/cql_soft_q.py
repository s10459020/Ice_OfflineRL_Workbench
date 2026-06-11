from dataclasses import dataclass

import math
import torch

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.sac import SACActor
from ice_offline.agent.sac import SACAgent
from ice_offline.agent.sac import SACCritic
from ice_offline.dataset._types import Batch


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

    def loss(self, conservative_loss_detached: torch.Tensor) -> torch.Tensor:
        # loss = -E[ alpha * L_cql ]
        # Lagrangian dual，若L_cql項長期偏大，則加強修改力度
        return -(self() * conservative_loss_detached).mean()


class _CQLCritic(SACCritic):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        threshold: float = 10.0,
        weight: float = 5.0,
        n_action_samples: int = 10,
    ):
        super().__init__(obs_size, act_size)
        self.threshold = threshold
        self.weight = weight
        self.n_action_samples = n_action_samples

    def eval_q_n(self, o: torch.Tensor, a_sample: torch.Tensor) -> torch.Tensor:
        o = o.repeat_interleave(self.n_action_samples, dim=0)  # (B, O) => (B*N, O)
        a = a_sample.reshape(-1, a_sample.shape[-1])           # (B*N, A)
        return torch.stack([q(o, a) for q in self.q_networks], dim=0)

    def eval_tq_n(self, o: torch.Tensor, a_sample: torch.Tensor) -> torch.Tensor:
        o = o.repeat_interleave(self.n_action_samples, dim=0)
        a = a_sample.reshape(-1, a_sample.shape[-1])
        return torch.stack([tq(o, a) for tq in self.tq_networks], dim=0)
    
    def shift_loss(self, loss_suppress: torch.Tensor) -> torch.Tensor:        
        return self.weight * (loss_suppress - self.threshold)


@dataclass
class CQLSoftQAgent(SACAgent):
    id: str = "cql_soft_q"
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
    # Update
    # ====================
    def update_critic(self, batch: Batch) -> None:
        loss_suppress = self.loss_suppress(batch)
        loss_suppress = self.critic.shift_loss(loss_suppress)
        # loss_multiplier = self.multiplier.loss(loss_suppress.detach())

        # Keep loss functions side-effect free; update CQL multiplier explicitly here.
        # self.multiplier.optimizer.zero_grad()
        # loss_multiplier.backward()
        # self.multiplier.optimizer.step()

        loss_critic = self.loss_critic_with_suppress(batch, loss_suppress)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        o, a, r, on, d = batch
        batch_size = o.shape[0]
        n = self.critic.n_action_samples

        loss_td = self.loss_td(batch)
        grad_td = self._grad_norm(loss_td, self.critic.parameters())

        a_s, logp = self.actor.sample_n(o, n)
        an, logpn = self.actor.sample_n(on, n)
        ar, logpr = self.actor.sample_random_n(o)

        q = self.critic.eval_q_n(o, a_s).view(2, batch_size, n)
        qn = self.critic.eval_q_n(o, an).view(2, batch_size, n)
        qr = self.critic.eval_q_n(o, ar).view(2, batch_size, n)
        q_cat = torch.cat([q, qn, qr], dim=2)

        logp = logp.view(1, batch_size, n)
        logpn = logpn.view(1, batch_size, n)
        logpr = logpr.view(1, batch_size, n)
        logp_cat = torch.cat([logp, logpn, logpr], dim=2)

        data_q = torch.stack(self.critic.q_all(o, a), dim=0)
        lse_q = torch.logsumexp(q_cat, dim=2, keepdim=True)
        lse_q_logp = torch.logsumexp(q_cat - logp_cat, dim=2, keepdim=True)
        loss_suppress = (lse_q_logp - data_q).mean(dim=[1, 2])
        loss_suppress_sum = loss_suppress.sum()
        grad_suppress = self._grad_norm(loss_suppress_sum, self.critic.parameters())

        loss_suppress_shifted = self.critic.shift_loss(loss_suppress)
        loss_critic = self.loss_critic_with_suppress(batch, loss_suppress_shifted)
        grad_critic = self._grad_norm(loss_critic, self.critic.parameters())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        a_policy, log_prob = self.actor.sample(o)
        loss_temperature = self.temp.loss(log_prob)
        grad_temperature = self._grad_norm(loss_temperature, self.temp.parameters())

        self.temp.optimizer.zero_grad()
        loss_temperature.backward()
        self.temp.optimizer.step()

        loss_actor = self.loss_actor_with_sample(batch, a_policy, log_prob)
        grad_actor = self._grad_norm(loss_actor, self.actor.parameters())

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        self.critic.update_target_soft()

        metrics = {
            "loss_td": loss_td.detach(),
            "grad_td": grad_td.detach(),
            "loss_suppress": loss_suppress_sum.detach(),
            "grad_suppress": grad_suppress.detach(),
            "loss_critic": loss_critic.detach(),
            "grad_critic": grad_critic.detach(),
            "loss_temperature": loss_temperature.detach(),
            "grad_temperature": grad_temperature.detach(),
            "loss_actor": loss_actor.detach(),
            "grad_actor": grad_actor.detach(),
            "q_data": data_q.mean().detach(),
            "q_sample": q_cat.mean().detach(),
            "q_policy_sample": q.mean().detach(),
            "q_next_sample": qn.mean().detach(),
            "q_random_sample": qr.mean().detach(),
            "lse_q": lse_q.mean().detach(),
            "lse_q_logp": lse_q_logp.mean().detach(),
            "gap_q_only": (lse_q - data_q).mean().detach(),
            "gap_q_logp": (lse_q_logp - data_q).mean().detach(),
            "logp_lift": (lse_q_logp - lse_q).mean().detach(),
            "logp_policy": logp.mean().detach(),
            "logp_next": logpn.mean().detach(),
            "logp_random": logpr.mean().detach(),
            "neg_logp_policy": (-logp).mean().detach(),
            "neg_logp_next": (-logpn).mean().detach(),
            "neg_logp_random": (-logpr).mean().detach(),
        }

        with torch.no_grad():
            ap, _ = self.actor.sample(o)
            ar, _ = self.actor.sample_random_n(o)
            or_ = o.repeat_interleave(n, dim=0)
            metrics.update({
                "multiplier": self.multiplier().item(),
                "temperature": self.temp().item(),
                "q_policy": self.critic.q_min(o, ap).mean().item(),
                "q_random": self.critic.q_min(or_, ar).mean().item(),
                "target_q": self.target_sac(on, r, d).mean().item(),
            })

        return metrics

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

        logp = logp.view(1, batch_size, self.critic.n_action_samples)   # (1,B,N)
        logpn = logpn.view(1, batch_size, self.critic.n_action_samples) # (1,B,N)
        logpr = logpr.view(1, batch_size, self.critic.n_action_samples) # (1,B,N)
        logp_cat = torch.cat([logp, logpn, logpr], dim=2)                # (1,B,3N)

        q = self.critic.eval_q_n(o, a_s).view(2, batch_size, self.critic.n_action_samples)
        qn = self.critic.eval_q_n(o, an).view(2, batch_size, self.critic.n_action_samples)
        qr = self.critic.eval_q_n(o, ar).view(2, batch_size, self.critic.n_action_samples)
        q_cat = torch.cat([q, qn, qr], dim=2)                            # (2,B,3N)

        logsumexp = torch.logsumexp(q_cat - logp_cat, dim=2, keepdim=True)  # (2,B,1)
        data_q = torch.stack(self.critic.q_all(o, a), dim=0)
        return (logsumexp - data_q).mean(dim=[1, 2])   

    def loss_critic_with_suppress(
        self,
        batch: Batch,
        loss_suppress: torch.Tensor,
    ) -> torch.Tensor:
        # CQL loss: loss_td + multiplier * loss_suppress
        loss_td = self.loss_critic(batch)
        # return loss_td + (self.multiplier() * loss_suppress).sum()
        return loss_td + (loss_suppress).sum()
