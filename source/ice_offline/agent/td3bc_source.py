from dataclasses import dataclass

import torch

from ice_offline.agent.td3 import TD3Agent
from ice_offline.dataset._spec import TorchBuffer


@dataclass
class TD3BCSourceAgent(TD3Agent):
    alpha: float = 2.5

    # ====================
    # Update
    # ====================
    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list
        r = batch.rew_list.view(-1, 1)
        on = batch.next_obs_list
        d = batch.done_list.view(-1, 1)

        self.update_critic(o, a, r, on, d)

        self.update_step += 1
        if self.update_step % self.update_actor_interval == 0:
            self.update_actor(o, a)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

    def update_actor(self, o: torch.Tensor, a: torch.Tensor) -> None:
        self.actor_optimizer.zero_grad()
        actor_loss = self.loss_actor(o, a)
        actor_loss.backward()
        self.actor_optimizer.step()

    # ====================
    # Actor mathmatics
    # ====================
    def loss_bc(self, a: torch.Tensor, a_pred: torch.Tensor) -> torch.Tensor:
        # loss = E{s,a~D}[ MSE(a - pi(s)) ]
        return ((a - a_pred) ** 2).mean()

    def loss_td3(self, o: torch.Tensor, a_pred: torch.Tensor) -> torch.Tensor:
        # source design
        q = self.critic.q_networks[0](o, a_pred)
        lam = self.alpha / q.abs().mean().detach()
        return lam * -q.mean()

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_pred = self.actor.pi(o)
        return self.loss_td3(o, a_pred) + self.loss_bc(a, a_pred)
