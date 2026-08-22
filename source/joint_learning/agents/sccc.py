import torch
from torch.nn import functional as F

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.scas import SCASAgent
from joint_learning.agents.variant import GPAgent, NAgent
from joint_learning.lib.dataset import Batch


class SCCCAgent(SCASAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        lambda_q: float = 5.0,
        k_sccc: int = 30,
        delta_q: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, **kwargs)
        self.lambda_q = lambda_q
        self.k_sccc = k_sccc
        self.delta_q = delta_q

    # ====================
    # Loss functions
    # ====================
    def loss_conservative(self, batch: Batch) -> torch.Tensor:
        # a\tilde_k \sim U(A), k = 1, ..., K_(SCCC)
        # Q_(LSE,i)(s) = log \sum_(k = 1)^(K_(SCCC)) exp(Q_i(s, a\tilde_k))
        # L_conservative = E_((s, a) \sim D) [
        #     \sum_i ReLU(Q_(LSE,i)(s) - Q_i(s, a) + \delta_Q)
        # ]
        observations, actions, _, _, _ = batch
        sampled_actions = self.actor.sample_uniform(observations, self.k_sccc)
        losses = []
        q_data_all = self.critic.q_all(observations, actions)
        q_sample_all = self.critic.q_all_n(observations, sampled_actions)
        for q_data, q_sample in zip(q_data_all, q_sample_all):
            q_lse = torch.logsumexp(q_sample, dim=1)
            losses.append(
                F.relu(q_lse - q_data + self.delta_q).mean()
            )
        return sum(losses)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # L_critic = L_TD + \lambda_Q L_conservative
        return self.loss_td(batch) + self.lambda_q * self.loss_conservative(batch)


class SCCCNAgent(NAgent, SCCCAgent):
    pass


class SCCCGPAgent(GPAgent, SCCCAgent):
    pass


class SCCCGPNAgent(GPAgent, SCCCNAgent):
    pass
