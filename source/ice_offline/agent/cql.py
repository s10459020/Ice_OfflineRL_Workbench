import math
import torch

from ice_offline.agent._spec import Agent
from ice_offline.agent._spec import MetricValues
from ice_offline.agent.sac import SACActor
from ice_offline.agent.sac import SACAgent
from ice_offline.agent.sac import SACCritic
from ice_offline.dataset._types import Batch


class _CQLActor(SACActor):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__(obs_size, act_size, config)
        self.act_size = act_size

    def sample_random_n(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = o.shape[0]
        action_shape = (batch_size, self.n_samples, self.act_size)
        prob_shape   = (batch_size, self.n_samples, 1)
        a = torch.zeros(action_shape,device=o.device,).uniform_(-1.0, 1.0)
        log_prob = torch.full(prob_shape, math.log(0.5**self.act_size), device=o.device)
        return a, log_prob


class _CQLMultiplier(torch.nn.Module):
    def __init__(self, config: dict[str, object] = {}):
        super().__init__()
        self.threshold = config.get("threshold", 1.0)
        tensor = torch.full((1, 1), math.log(config.get("scale_init", 10.0)), dtype=torch.float32)
        self.log_scale = torch.nn.Parameter(tensor)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(0.0, 1e6)

    def loss(self, loss_suppress: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gap = loss_suppress - self.threshold
        loss = -(self() * gap).mean()
        return loss, {
            "loss_multiplier": Agent._value(loss.detach()),
            "grad_multiplier": Agent._grad_norm(loss, self.parameters()),
            "multiplier": Agent._value(self().detach()),
        }


class _CQLCritic(SACCritic):
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

class CQLAgent(SACAgent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.actor = _CQLActor(self.obs_size, self.act_size, config).to(self.device)
        self.critic = _CQLCritic(self.obs_size, self.act_size, config).to(self.device)
        self.multiplier = _CQLMultiplier(config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.param_critic())

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_suppress",
            "grad_suppress",
            "loss_critic",
            "grad_critic",
            "loss_sac",
            "grad_sac",
            "loss_temp",
            "grad_temp",
            "loss_multiplier",
            "grad_multiplier",
            "temp",
            "multiplier",
            "target_q",
            "q_cat",
            "logp_cat",
            "logsumexp",
            "grad_logsumexp",
            "data_q",
            "grad_data_q",
        ]

    def update(self, batch: Batch) -> MetricValues:
        metrics = self.update_critic(batch)
        metrics |= self.update_actor(batch)
        metrics |= self.update_temperature(batch)
        self.critic.update_target_soft()
        return metrics

    def update_critic(self, batch: Batch) -> MetricValues:
        loss_td, metrics_td = self.loss_td(batch)
        loss_suppress, metrics_suppress = self.loss_suppress(batch)
        
        metrics_multiplier = self.update_multiplier(loss_suppress)

        loss_critic = loss_td + (self.multiplier().detach() * loss_suppress)
        grad_critic = self._grad_norm(loss_critic, self.critic.param_critic())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        return metrics_td | {
            "loss_suppress": self._value(loss_suppress.detach()),
            "grad_suppress": metrics_suppress["grad_suppress"],
            "loss_critic": self._value(loss_critic.detach()),
            "grad_critic": grad_critic,
        } | metrics_suppress | metrics_multiplier

    def update_multiplier(self, loss_suppress: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_multiplier, metrics = self.multiplier.loss(loss_suppress.detach())
        self.multiplier.optimizer.zero_grad()
        loss_multiplier.backward()
        self.multiplier.optimizer.step()
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
    def _log_prob_n(self, log_prob: torch.Tensor) -> torch.Tensor:
        # (B, N, 1) -> (1, B, N)
        return log_prob.transpose(1, 2).transpose(0, 1)
    
    def loss_suppress(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        grad_logsumexp = self._grad_norm(logsumexp.mean(), self.critic.param_critic())
        data_q = torch.stack(self.critic.q_all(o, a), dim=0)
        grad_data_q = self._grad_norm(data_q.mean(), self.critic.param_critic())

        loss = (logsumexp - data_q).mean()

        return loss, {
            "q_cat": self._value(q_cat.mean().detach()),
            "logp_cat": self._value(logp_cat.mean().detach()),
            "logsumexp": self._value(logsumexp.mean().detach()),
            "grad_logsumexp": grad_logsumexp,
            "data_q": self._value(data_q.mean().detach()),
            "grad_data_q": grad_data_q,
            "grad_suppress": self._grad_norm(loss, self.critic.param_critic()),
        }
