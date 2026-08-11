import torch
from torch.nn import functional as F

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scas import SCASAgent
from joint_learning.lib.dataset import Batch


class SCASPLAgent(SCASAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamics: SCASDynamics,
        weight_correction: float = 0.25,
        weight_punish: float = 2.5,
        actor_num_sample: int = 16,
        critic_rate_decay: float = 0.005,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            obs_size,
            act_size,
            dynamics=dynamics,
            weight_correction=weight_correction,
            device=device,
        )
        self.weight_punish = weight_punish
        self.actor_num_sample = actor_num_sample
        self.critic_rate_decay = critic_rate_decay
        self.q_avg = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def update(self, batch: Batch) -> None:
        self.update_step += 1
        loss_critic = self.loss_critic(batch)
        self.q_optimizer.zero_grad()
        loss_critic.backward()
        self.q_optimizer.step()

        if self.update_step % self.update_actor_interval == 0:
            loss_actor = self.loss_actor(batch)
            self.policy_optimizer.zero_grad()
            loss_actor.backward()
            self.policy_optimizer.step()
            self.sync_target_soft()

    def sample_actions_uniform(self, batch_size: int) -> torch.Tensor:
        return torch.empty(
            (self.actor_num_sample, batch_size, self.act_size),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-self.max_action, self.max_action)

    def action_distance(self, actions: torch.Tensor, sampled_actions: torch.Tensor) -> torch.Tensor:
        diff = (actions - sampled_actions) ** 2
        return (diff / ((2 * self.max_action) ** 2)).mean(dim=2, keepdim=True)

    def update_q_avg(self, q_anchor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current = q_anchor.abs().mean()
            if self.q_avg.item() == 0.0:
                self.q_avg.copy_(current)
            else:
                self.q_avg.mul_(1.0 - self.critic_rate_decay)
                self.q_avg.add_(self.critic_rate_decay * current)
        return self.q_avg

    def loss_punish(self, batch: Batch) -> torch.Tensor:
        # SCASPL pseudo-label punishment:
        #   Q~(s, a_sample) = Q_target(s, a_data) - c * d(a_data, a_sample)
        #   L_PL = E[(Q_i(s, a_sample) - Q~(s, a_sample))^2].
        observations, actions, _, _, _ = batch
        sampled_actions = self.sample_actions_uniform(observations.shape[0])
        distance = self.action_distance(actions, sampled_actions)

        with torch.no_grad():
            q_anchor = torch.minimum(
                self.target_q1(observations, actions),
                self.target_q2(observations, actions),
            )
            q_pseudo = q_anchor.unsqueeze(0) - self.update_q_avg(q_anchor) * distance

        flat_observations = observations.unsqueeze(0).expand(sampled_actions.shape[0], -1, -1)
        flat_observations = flat_observations.reshape(-1, self.obs_size)
        flat_actions = sampled_actions.reshape(-1, self.act_size)
        flat_pseudo = q_pseudo.reshape(-1, 1)

        return F.mse_loss(self.q1(flat_observations, flat_actions), flat_pseudo) + F.mse_loss(
            self.q2(flat_observations, flat_actions),
            flat_pseudo,
        )

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # SCASPL critic loss:
        #   L_Q = L_TD + lambda_p * L_PL.
        return self.loss_td(batch) + self.weight_punish * self.loss_punish(batch)
