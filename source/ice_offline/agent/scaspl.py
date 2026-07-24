import torch
import torch.nn.functional as F

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.aspl import AsplAgent
from ice_offline.agent.aspl import AsplCritic
from ice_offline.agent.scas import ScasAgent
from ice_offline.agent.td3 import TD3Actor
from ice_offline.dataset._types import Batch


class ScasplActor(TD3Actor):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__(obs_size, act_size, config)
        self.num_sample = config.get("actor_num_sample", 16)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(int(seed))

    def sample_actions_uniform(self, batch_size: int) -> torch.Tensor:
        return torch.empty(
            (self.num_sample, batch_size, self.act_size),
            dtype=torch.float32,
            device=next(self.pi.parameters()).device,
        ).uniform_(-self.max_action, self.max_action)

    def action_distance(self, a: torch.Tensor, a_samples: torch.Tensor) -> torch.Tensor:
        diff = (a - a_samples) ** 2
        normalize = (2 * self.max_action) ** 2
        result = diff / normalize
        return result.mean(dim=2, keepdim=True)


class ScasplAgent(ScasAgent, AsplAgent):
    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_punish = config.get("weight_punish", 2.5)
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config=config,
            device=device,
        )
        self.actor = ScasplActor(self.obs_size, self.act_size, config).to(self.device)
        self.critic = AsplCritic(self.obs_size, self.act_size, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.param_actor())
        self.critic_optimizer = torch.optim.Adam(self.critic.param_critic())
        self.dynamics = self.dynamics.prepare()

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "q_avg",
            "target_q",
        ]

    def update(self, batch: Batch) -> MetricValues:
        metrics = self.update_critic(batch)
        metrics["q_avg"] = self._value(self.critic.q_avg.detach())

        self.update_step += 1
        if self.update_step % self.update_actor_interval == 0:
            metrics |= self.update_actor(batch)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

        return metrics

    def loss_punish(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s, a, _, _, _ = batch

        a_samples = self.actor.sample_actions_uniform(s.shape[0])
        action_distance = self.actor.action_distance(a, a_samples)
        q_pseudo = self.critic.q_pseudo(s, a, action_distance)

        s_reshape = s.unsqueeze(0).expand(a_samples.shape[0], -1, -1).reshape(-1, s.shape[1])
        a_samples_reshape = a_samples.view(-1, a.shape[1])
        q_pseudo_reshape = q_pseudo.view(-1, 1)

        q_values = (
            self.critic.q_networks[0](s_reshape, a_samples_reshape),
            self.critic.q_networks[1](s_reshape, a_samples_reshape),
        )
        loss = sum(F.mse_loss(q_value, q_pseudo_reshape) for q_value in q_values)
        return loss, {
            "loss_punish": self._value(loss.detach()),
            "grad_punish": self._grad_norm(loss, self.critic.param_critic()),
        }
