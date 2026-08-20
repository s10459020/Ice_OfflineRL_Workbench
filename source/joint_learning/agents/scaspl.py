import torch
from torch.nn import functional as F

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.scas import SCASAgent
from joint_learning.agents.variant import CAgent, GPAgent, NAgent
from joint_learning.lib.dataset import Batch


class SCASPLAgent(SCASAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        lambda_q: float = 0.05,
        k_scaspl: int = 6,
        rho_c: float = 0.005,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, **kwargs)
        self.lambda_q = lambda_q
        self.k_scaspl = k_scaspl
        self.rho_c = rho_c
        self.c_t = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=self.device,
        )

    # ====================
    # Help functions
    # ====================
    def action_distance(self, actions: torch.Tensor, sampled_actions: torch.Tensor) -> torch.Tensor:
        # d(a, a\tilde) = (1/A)\sum_j ((a_j - a\tilde_j)/(2a_(max)))^2
        diff = (actions.unsqueeze(1) - sampled_actions) ** 2
        return (diff / ((2 * self.max_action) ** 2)).mean(dim=2, keepdim=True)

    def update_c_t(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # c_t = (1 - \rho_c)c_(t - 1) + \rho_c E_((s, a) \sim D) [(|Q_1 (s, a)| + |Q_2 (s, a)|)/2]
        with torch.no_grad():
            q_values = self.critic.q_all(observations, actions)
            current = torch.stack(q_values).abs().mean()
            if self.c_t.item() == 0.0:
                self.c_t.copy_(current)
            else:
                self.c_t.mul_(1.0 - self.rho_c)
                self.c_t.add_(self.rho_c * current)
        return self.c_t

    def update(self, batch: Batch) -> None:
        observations, actions, _, _, _ = batch
        self.update_c_t(observations, actions)
        super().update(batch)

    # ====================
    # Loss functions
    # ====================
    def loss_pseudo(self, batch: Batch) -> torch.Tensor:
        # Q\tilde (s, a\tilde_k) = min_i Q_i^target (s, a) - c_t d(a, a\tilde_k)
        # Loss_pseudo = E_((s, a) \sim D) [(1/K_(SCASPL))\sum_k \sum _(i = 1)^2 (Q_i (s, a\tilde_k) - Q\tilde (s, a\tilde_k))^2]
        observations, actions, _, _, _ = batch
        sampled_actions = self.actor.sample_uniform(observations, self.k_scaspl)
        distance = self.action_distance(actions, sampled_actions)

        with torch.no_grad():
            q_target = self.critic.t_min(observations, actions)
            q_pseudo = q_target.unsqueeze(1) - self.c_t * distance

        q_values = self.critic.q_all_n(observations, sampled_actions)
        return sum(F.mse_loss(q, q_pseudo) for q in q_values)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Q^SCASPL = Loss_TD + \lambda_Q Loss_pseudo
        return self.loss_td(batch) + self.lambda_q * self.loss_pseudo(batch)


class SCASPLNAgent(NAgent, SCASPLAgent):
    pass


class SCASPLGPAgent(GPAgent, SCASPLAgent):
    pass


class SCASPLCAgent(CAgent, SCASPLAgent):
    pass


class SCASPLNCAgent(CAgent, SCASPLNAgent):
    pass
