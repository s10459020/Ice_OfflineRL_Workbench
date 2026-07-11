import torch
import torch.nn.functional as F

from ice_offline.agent.scaspl_n import ScasplNAgent
from ice_offline.dataset._types import Batch


class ScasplNSAgent(ScasplNAgent):
    def loss_punish(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s, a, r, sn, d = batch
        target = self.target_td3(sn, r, d)

        a_samples = self.actor.sample_actions_lhs(s.shape[0])
        action_distance = self.actor.action_distance(a, a_samples)
        q_pseudo = self.critic.q_pseudo(target, action_distance)

        s_noise = self.dynamics.noise_state(s)
        s_noise_reshape = s_noise.unsqueeze(0).expand(a_samples.shape[0], -1, -1).reshape(-1, s.shape[1])
        a_samples_reshape = a_samples.view(-1, a.shape[1])
        q_pseudo_reshape = q_pseudo.view(-1, 1)

        q_values = (
            self.critic.q_networks[0](s_noise_reshape, a_samples_reshape),
            self.critic.q_networks[1](s_noise_reshape, a_samples_reshape),
        )
        losses = [F.mse_loss(q_value, q_pseudo_reshape) for q_value in q_values]
        loss = sum(losses)
        return loss, {
            "loss_punish": self._value(loss.detach()),
            "grad_punish": self._grad_norm(loss, self.critic.param_critic()),
        }
