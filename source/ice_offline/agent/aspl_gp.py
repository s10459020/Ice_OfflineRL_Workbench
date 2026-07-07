import torch
import torch.nn.functional as F

from ice_offline.agent.aspl import AsplAgent
from ice_offline.dataset._types import Batch


class AsplGPAgent(AsplAgent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.weight_gp = config.get("weight_gp", 1.0)
        self.gp_threshold = config.get("gp_threshold", 1.0)
        self.gp_count = config.get("gp_count", 16)

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_gp",
            "grad_gp",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "q_avg",
            "target_q",
            "grad_norm",
        ]

    def loss_gp(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, _, _, _, _ = batch
        o_gp = o.repeat_interleave(self.gp_count, dim=0).detach()
        a_gp = self.actor.sample_random_n(o, self.gp_count).reshape(-1, self.act_size)
        a_gp.requires_grad_(True)
        q_values = self.critic.q_all(o_gp, a_gp)

        penalties = []
        for q in q_values:
            grad = torch.autograd.grad(
                outputs=q.sum(),
                inputs=a_gp,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_norm = grad.norm(p=2, dim=-1)
            penalties.append(F.relu(grad_norm - self.gp_threshold).square())

        loss = torch.stack(penalties, dim=0).sum(dim=0).mean()
        return loss, {
            "grad_norm": self._value(grad_norm.detach().mean()),
            "loss_gp": self._value(loss.detach()),
            "grad_gp": self._grad_norm(loss, self.critic.param_critic()),
        }

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_aspl, metrics_aspl = super().loss_critic(batch)
        loss_gp, metrics_gp = self.loss_gp(batch)
        loss = loss_aspl + self.weight_gp * loss_gp
        return loss, metrics_aspl | metrics_gp | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
