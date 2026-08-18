import torch
from torch.nn import functional as F

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.scas import SCASAgent
from joint_learning.lib.dataset import Batch


class SCASGPAgent(SCASAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        lambda_gp: float = 1.0,
        gp_count: int = 16,
        gp_threshold: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)
        self.lambda_gp = lambda_gp
        self.gp_count = gp_count
        self.gp_threshold = gp_threshold

    # -------------------------------------------------------------------------
    # Help functions
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_gradient(self, batch: Batch) -> torch.Tensor:
        # g_i = ||\nabla _(a\hat) Q_i (s,a\hat)||_2
        # Loss_gradient = E_(s\sim D,a\hat\sim U(A)) [\sum _(i=1)^2 ReLU(g_i-\delta_(GP))^2]
        sample_count = self.gp_count
        threshold = self.gp_threshold
        observations, _, _, _, _ = batch
        batch_size = observations.shape[0]
        sampled_actions = self.actor.sample_uniform(observations, sample_count)
        flat_actions = sampled_actions.reshape(batch_size * sample_count, self.act_size).requires_grad_(True)
        flat_observations = observations.unsqueeze(1).expand(-1, sample_count, -1)
        flat_observations = flat_observations.reshape(batch_size * sample_count, self.obs_size).detach()
        penalties = []
        for grad in self.critic.action_gradients(flat_observations, flat_actions):
            penalties.append(F.relu(grad.norm(p=2, dim=-1) - threshold).square())
        return torch.stack(penalties, dim=0).sum(dim=0).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_TD + \lambda_(GP) Loss_gradient
        return self.loss_td(batch) + self.lambda_gp * self.loss_gradient(batch)
