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

    # -------------------------------------------------------------------------
    # Help functions
    # -------------------------------------------------------------------------
    def sample_actions_uniform(self, batch_size: int) -> torch.Tensor:
        # a\tilde_k \sim U([-a_(max), a_(max)]A), k \in {1, ..., K}
        return torch.empty(
            (self.actor_num_sample, batch_size, self.act_size),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-self.max_action, self.max_action)

    def action_distance(self, actions: torch.Tensor, sampled_actions: torch.Tensor) -> torch.Tensor:
        # d(a,a\tilde)=(1/A)\sum_j ((a_j - a\tilde_j)/(2a_(max)))^2
        diff = (actions - sampled_actions) ** 2
        return (diff / ((2 * self.max_action) ** 2)).mean(dim=2, keepdim=True)

    def update_q_avg(self, q_hat_1: torch.Tensor, q_hat_2: torch.Tensor) -> torch.Tensor:
        # c_t=(1-\rho)c_(t-1)+\rho E_D [(|Q_1(s,a)|+|Q_2(s,a)|)/2]
        with torch.no_grad():
            current = 0.5 * (q_hat_1.abs().mean() + q_hat_2.abs().mean())
            if self.q_avg.item() == 0.0:
                self.q_avg.copy_(current)
            else:
                self.q_avg.mul_(1.0 - self.critic_rate_decay)
                self.q_avg.add_(self.critic_rate_decay * current)
        return self.q_avg

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_pseudo_label_constraint(self, batch: Batch) -> torch.Tensor:
        # Q\tilde(s,a\tilde_k)=min_i sg(Q_i^target(s,a))-c_t d(a,a\tilde_k)
        # Loss_pseudo = E_D [(1/K) * \sum_k \sum_(i=1)^2 (Q_i(s,a\tilde_k)-Q\tilde(s,a\tilde_k))^2]
        observations, actions, _, _, _ = batch
        sampled_actions = self.sample_actions_uniform(observations.shape[0])
        distance = self.action_distance(actions, sampled_actions)

        q_hat_1 = self.q1(observations, actions)
        q_hat_2 = self.q2(observations, actions)
        with torch.no_grad():
            q_anchor = torch.minimum(
                self.target_q1(observations, actions),
                self.target_q2(observations, actions),
            )
            q_pseudo = q_anchor.unsqueeze(0) - self.update_q_avg(q_hat_1, q_hat_2) * distance

        flat_observations = observations.unsqueeze(0).expand(sampled_actions.shape[0], -1, -1)
        flat_observations = flat_observations.reshape(-1, self.obs_size)
        flat_actions = sampled_actions.reshape(-1, self.act_size)
        flat_pseudo = q_pseudo.reshape(-1, 1)

        return F.mse_loss(self.q1(flat_observations, flat_actions), flat_pseudo) + F.mse_loss(
            self.q2(flat_observations, flat_actions),
            flat_pseudo,
        )

    def loss_q(self, batch: Batch) -> torch.Tensor:
        # SCASPL critic loss:
        # Loss_Q = Loss_TD + \lambda_p * Loss_Pseudo_Label_Constraint
        return self.loss_td(batch) + self.weight_punish * self.loss_pseudo_label_constraint(batch)
