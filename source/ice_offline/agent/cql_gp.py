import torch
import torch.nn.functional as F

from ice_offline.agent.cql import CQLAgent
from ice_offline.dataset._types import Batch


class CQLGPAgent(CQLAgent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.weight_gp = config.get("weight_gp", 1.0)
        self.gp_threshold = config.get("gp_threshold", 1.0)
        self.gp_count = config.get("gp_count", 16)

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_suppress",
            "grad_suppress",
            "loss_gp",
            "grad_gp",
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
            "grad_norm",
        ]

    def update(self, batch: Batch) -> dict[str, torch.Tensor]:
        metrics = self.update_critic(batch)
        metrics |= self.update_actor(batch)
        metrics |= self.update_temperature(batch)
        self.critic.update_target_soft()
        return metrics

    def update_critic(self, batch: Batch) -> dict[str, torch.Tensor]:
        loss_td, metrics_td = self.loss_td(batch)
        loss_suppress, metrics_suppress = self.loss_suppress(batch)
        loss_gp, metrics_gp = self.loss_gp(batch)
        metrics_multiplier = self.update_multiplier(loss_suppress)

        loss_critic = (
            loss_td
            + (self.multiplier().detach() * loss_suppress)
            + self.weight_gp * loss_gp
        )
        grad_critic = self._grad_norm(loss_critic, self.critic.param_critic())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        return metrics_td | {
            "loss_suppress": self._value(loss_suppress.detach()),
            "grad_suppress": metrics_suppress["grad_suppress"],
            "loss_critic": self._value(loss_critic.detach()),
            "grad_critic": grad_critic,
        } | metrics_suppress | metrics_gp | metrics_multiplier

    # ====================
    # Critic loss
    # ====================
    def loss_gp(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, _, _, _, _ = batch
        o_gp = o.repeat_interleave(self.gp_count, dim=0).detach()
        a_gp = torch.empty(
            (o.shape[0], self.gp_count, self.act_size),
            device=o.device,
            dtype=o.dtype,
        ).uniform_(-1.0, 1.0).reshape(-1, self.act_size)
        a_gp.requires_grad_(True)
        q_values = self.critic.q_all(o_gp, a_gp)

        penalties = []
        grad_norm_mean = torch.zeros((), device=o.device)
        for q in q_values:
            grad = torch.autograd.grad(
                outputs=q.sum(),
                inputs=a_gp,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_norm = grad.norm(p=2, dim=-1)
            penalties.append(F.relu(grad_norm - self.gp_threshold).square())
            grad_norm_mean = grad_norm_mean + grad_norm.mean()

        grad_norm_mean = grad_norm_mean / len(q_values)
        loss = torch.stack(penalties, dim=0).sum(dim=0).mean()
        return loss, {
            "grad_norm": self._value(grad_norm_mean.detach()),
            "loss_gp": self._value(loss.detach()),
            "grad_gp": self._grad_norm(loss, self.critic.param_critic()),
        }
