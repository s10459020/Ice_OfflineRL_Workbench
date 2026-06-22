from dataclasses import dataclass
import math
import torch

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.sac import SACActor
from ice_offline.agent.sac import SACAgent
from ice_offline.agent.sac import SACCritic
from ice_offline.dataset._types import Batch


class _CQLActor(SACActor):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__(obs_size, act_size)
        self.act_size = act_size

    def sample_random_n(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = o.shape[0]
        action_shape = (batch_size, self.n_samples, self.act_size)
        prob_shape   = (batch_size, self.n_samples, 1)
        a = torch.zeros(action_shape,device=o.device,).uniform_(-1.0, 1.0)
        log_prob = torch.full(prob_shape, math.log(0.5**self.act_size), device=o.device)
        return a, log_prob


class _CQLMultiplier(torch.nn.Module):
    def __init__(self, 
            learning_rate: float = 1e-4, 
            scale_init: float = 10.0, 
            threshold: float = 2, 
        ):
        super().__init__()
        self.threshold = threshold

        tensor = torch.full((1, 1), math.log(scale_init), dtype=torch.float32)
        self.log_scale = torch.nn.Parameter(tensor)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(0.0, 1e6)

    def loss(self, loss_suppress: torch.Tensor) -> torch.Tensor:
        gap = loss_suppress - self.threshold
        return -(self() * gap).mean()


class _CQLCritic(SACCritic):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__(obs_size, act_size)

    def eval_q_n(self, o: torch.Tensor, a_sample: torch.Tensor) -> torch.Tensor:
        batch_size = o.shape[0]
        n = a_sample.shape[1]
        o = o.repeat_interleave(n, dim=0)             # (B, O) -> (B*N, O)
        a = a_sample.reshape(-1, a_sample.shape[-1])  # (B, N, A) -> (B*N, A)
        q_values = torch.stack([q(o, a) for q in self.q_networks], dim=0)
        # (Q, B*N, 1) -> (Q, B, N, 1)
        return q_values.view(len(self.q_networks), batch_size, n, 1)

    def eval_tq_n(self, o: torch.Tensor, a_sample: torch.Tensor) -> torch.Tensor:
        batch_size = o.shape[0]
        n = a_sample.shape[1]
        o = o.repeat_interleave(n, dim=0)
        a = a_sample.reshape(-1, a_sample.shape[-1])
        tq_values = torch.stack([tq(o, a) for tq in self.tq_networks], dim=0)
        # (Q, B*N, 1) -> (Q, B, N, 1)
        return tq_values.view(len(self.tq_networks), batch_size, n, 1)

@dataclass
class CQLAgent(SACAgent):
    id: str = "cql"
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 1e-4
    multiplier_learning_rate: float = 1e-4
    scale_init: float = 10.0
    threshold: float = 1.0
    update_step = 0

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        super().__post_init__()
        self.actor = _CQLActor(obs_size=self.obs_size, act_size=self.act_size).to(self.device)
        self.critic = _CQLCritic(obs_size=self.obs_size, act_size=self.act_size).to(self.device)
        self.multiplier = _CQLMultiplier(
            learning_rate=self.multiplier_learning_rate,
            scale_init=self.scale_init,
            threshold=self.threshold,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.pi.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.q_networks.parameters(), lr=self.critic_learning_rate)

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch):
        self.update_critic(batch)
        self.update_actor(batch)
        self.update_temperature(batch)
        self.critic.update_target_soft()

    def update_actor_with_metrics(self, batch: Batch) -> dict:
        loss_actor = self.loss_actor(batch)
        grad_actor = self._grad_norm(loss_actor, self.actor.parameters())

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        return {
            "loss_actor": loss_actor.detach(),
            "grad_actor": grad_actor.detach(),
        }

    def update_critic_with_metrics(self, batch: Batch) -> dict:
        loss_td = self.loss_critic(batch)
        grad_td = self._grad_norm(loss_td, self.critic.parameters())

        loss_suppress, metrics_suppress = self.update_suppress_with_metrics(batch)
        grad_suppress = self._grad_norm(loss_suppress, self.critic.parameters())

        metrics_multiplier = self.update_multiplier_with_metrics(loss_suppress)
        
        loss_critic = loss_td + (self.multiplier().detach() * loss_suppress)
        grad_critic = self._grad_norm(loss_critic, self.critic.parameters())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        return {
            "loss_td": loss_td.detach(),
            "grad_td": grad_td.detach(),
            "loss_suppress": loss_suppress.detach(),
            "grad_suppress": grad_suppress.detach(),
            "loss_critic": loss_critic.detach(),
            "grad_critic": grad_critic.detach(),
        } | metrics_suppress | metrics_multiplier

    def update_suppress_with_metrics(self, batch: Batch) -> tuple[torch.Tensor, dict]:
        o, a, _, on, _ = batch

        with torch.no_grad():
            a_s, logp = self.actor.sample_n(o)
            an, logpn = self.actor.sample_n(on)
            ar, logpr = self.actor.sample_random_n(o)

            logp = self._log_prob_n(logp)
            logpn = self._log_prob_n(logpn)
            logpr = self._log_prob_n(logpr)
            logp_cat = torch.cat([logp, logpn, logpr], dim=2)  # (1, B, 3N)

        q = self.critic.eval_q_n(o, a_s).squeeze(-1)
        qn = self.critic.eval_q_n(o, an).squeeze(-1)
        qr = self.critic.eval_q_n(o, ar).squeeze(-1)
        q_cat = torch.cat([q, qn, qr], dim=2)  # (2, B, 3N)

        logsumexp = torch.logsumexp(q_cat - logp_cat, dim=2, keepdim=True)  # (2, B, 1)
        grad_logsumexp = self._grad_norm(logsumexp.mean(), self.critic.parameters())
        data_q = torch.stack(self.critic.q_all(o, a), dim=0)
        grad_data_q = self._grad_norm(data_q.mean(), self.critic.parameters())

        loss = (logsumexp - data_q).mean()

        return loss, {
            "q_cat": q_cat.mean().detach(),
            "logp_cat": logp_cat.mean().detach(),
            "logsumexp": logsumexp.mean().detach(),
            "grad_logsumexp": grad_logsumexp.detach(),
            "data_q": data_q.mean().detach(),
            "grad_data_q": grad_data_q.detach(),
        }

    def update_multiplier_with_metrics(self, loss_suppress: torch.Tensor) -> dict:
        loss = self.multiplier.loss(loss_suppress.detach())
        grad = self._grad_norm(loss, self.multiplier.parameters())

        self.multiplier.optimizer.zero_grad()
        loss.backward()
        self.multiplier.optimizer.step()
        return {
            "loss_multiplier": loss.detach(), 
            "grad_multiplier": grad.detach(),
            "multiplier": self.multiplier().detach(),
        }
    
    def update_temp_with_metrics(self, batch: Batch) -> dict:
        o, _, _, _, _ = batch
        _, log_prob = self.actor.sample(o)
        loss = self.temp.loss(log_prob)
        grad = self._grad_norm(loss, self.temp.parameters())

        self.temp.optimizer.zero_grad()
        loss.backward()
        self.temp.optimizer.step()
        return {
            "temp": self.temp().detach(),
            "loss_temp": loss.detach(),
            "grad_temp": grad.detach(),
        }

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        metrics_critic = self.update_critic_with_metrics(batch)
        metrics_actor = self.update_actor_with_metrics(batch)
        metrics_temp = self.update_temp_with_metrics(batch)
        self.critic.update_target_soft()
        return metrics_actor | metrics_critic | metrics_temp

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
    def _log_prob_n(self, log_prob: torch.Tensor) -> torch.Tensor:
        # (B, N, 1) -> (1, B, N)
        return log_prob.transpose(1, 2).transpose(0, 1)
    
    def loss_suppress(self, batch: Batch) -> torch.Tensor:
        # L_CQL(H) = E_D[s]{log *              sum_a[exp(Q)] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         sum_a[p * exp(Q) / p] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         E_a[exp(Q - log p)] } - E_D[s,a]{Q}
        #         ~= E_D[s]{log * 1/N * sum_N[exp(Q - log p)] } - E_D[s,a]{Q}
        #          = logsumexp(Q - log p) - E_(s,a)~D[Q]
        #
        # E_D[s]: input o
        # E_D[s,a]: input o,a
        # CQL sample approximation: a ~ p(a) => Uniform / pi(.|s) / pi(.|s')，各取 N 次
        o, a, _, on, _ = batch

        a_s, logp = self.actor.sample_n(o)
        an, logpn = self.actor.sample_n(on)
        ar, logpr = self.actor.sample_random_n(o)

        logp = self._log_prob_n(logp)
        logpn = self._log_prob_n(logpn)
        logpr = self._log_prob_n(logpr)
        logp_cat = torch.cat([logp, logpn, logpr], dim=2)  # (1, B, 3N)

        q = self.critic.eval_q_n(o, a_s).squeeze(-1)
        qn = self.critic.eval_q_n(o, an).squeeze(-1)
        qr = self.critic.eval_q_n(o, ar).squeeze(-1)
        q_cat = torch.cat([q, qn, qr], dim=2)  # (2, B, 3N)

        logsumexp = torch.logsumexp(q_cat - logp_cat, dim=2, keepdim=True)  # (2, B, 1)
        data_q = torch.stack(self.critic.q_all(o, a), dim=0)
        return (logsumexp - data_q).mean()   

    def loss_critic_with_suppress(
        self,
        batch: Batch,
        loss_suppress: torch.Tensor,
    ) -> torch.Tensor:
        # CQL loss: loss_td + multiplier * loss_suppress
        loss_td = self.loss_critic(batch)
        return loss_td + (self.multiplier() * loss_suppress).sum()
        # return loss_td + (loss_suppress).sum()
