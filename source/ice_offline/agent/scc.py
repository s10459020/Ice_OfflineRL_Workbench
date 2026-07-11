import math

import torch
import torch.nn.functional as F

from ice_offline.agent.scas import ScasAgent
from ice_offline.dataset._types import Batch


class SccAgent(ScasAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(obs_size=obs_size, act_size=act_size, dynamics=dynamics, config=config, device=device)
        self.weight_conservative = config.get("weight_conservative", config.get("alpha", 10))
        self.conservative_count = int(config.get("conservative_count", config.get("noise_count", 16)))
        self.threshold = config.get("threshold", config.get("threshold", 2.0))

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_conservative",
            "grad_conservative",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "target_q",
        ]

    def loss_conservative(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # loss = E[ logsumexp(Q(s,a~)) - log(N) - Q(s,a) ]
        o, a_data, _, _, _ = batch
        batch_size = o.shape[0]
        noise_count = self.conservative_count

        a_noise = self.actor.sample_random_n(o, noise_count)
        o_repeat = o.unsqueeze(1).expand(-1, noise_count, -1).reshape(batch_size * noise_count, self.obs_size)
        a_noise = a_noise.reshape(batch_size * noise_count, self.act_size)

        losses = []
        for q_network in self.critic.q_networks:
            q_data = q_network(o, a_data)
            q_noise = q_network(o_repeat, a_noise).view(batch_size, noise_count, 1)
            p_value = torch.logsumexp(q_noise, dim=1) - math.log(noise_count)
            losses.append(F.relu(p_value - q_data + self.threshold).mean())

        loss = sum(losses)
        return loss, {
            "loss_conservative": self._value(loss.detach()),
            "grad_conservative": self._grad_norm(loss, self.critic.param_critic()),
        }

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_td, metrics_td = self.loss_td(batch)
        loss_conservative, metrics_conservative = self.loss_conservative(batch)
        loss = loss_td + self.weight_conservative * loss_conservative
        return loss, metrics_td | metrics_conservative | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
