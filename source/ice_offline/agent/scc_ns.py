import math

import torch
import torch.nn.functional as F

from ice_offline.agent.scc_n import SccNAgent
from ice_offline.dataset._types import Batch


class SccNSAgent(SccNAgent):
    def loss_conservative(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # loss = E[ logsumexp(Q(s_noise,a~)) - log(N) - Q(s_noise,a) ]
        o, a_data, _, _, _ = batch
        batch_size = o.shape[0]
        noise_count = self.conservative_count

        o_noise = self.dynamics.noise_state(o)
        a_noise = self.actor.sample_random_n(o, noise_count)
        o_noise_repeat = o_noise.unsqueeze(1).expand(-1, noise_count, -1).reshape(batch_size * noise_count, self.obs_size)
        a_noise = a_noise.reshape(batch_size * noise_count, self.act_size)

        losses = []
        for q_network in self.critic.q_networks:
            q_data = q_network(o_noise, a_data)
            q_noise = q_network(o_noise_repeat, a_noise).view(batch_size, noise_count, 1)
            p_value = torch.logsumexp(q_noise, dim=1) - math.log(noise_count)
            losses.append(F.relu(p_value - q_data + self.threshold).mean())

        loss = sum(losses)
        return loss, {
            "loss_conservative": self._value(loss.detach()),
            "grad_conservative": self._grad_norm(loss, self.critic.param_critic()),
        }
