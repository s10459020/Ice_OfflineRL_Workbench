"""Conservative Q-Learning discrete agent (minimal fixed structure)."""

import numpy as np
import torch
import torch.nn.functional as F


class _Q(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.network(obs_t)


class CQLAgentDiscrete:
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        learning_rate: float = 6.25e-5,
        gamma: float = 0.99,
        alpha: float = 1.0,
        target_update_interval: int = 8000,
    ):
        self.device = "cpu"
        self.gamma = gamma
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.action_size = act_size

        self.q_network = _Q(obs_size, act_size).to(self.device)
        self.target_q_network = _Q(obs_size, act_size).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Public API
    # ====================
    def action_best_batch(self, obs_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_t = self.q_network(obs_t)
            action = q_t.argmax(dim=1).long()
        return action.cpu().numpy()

    def action_best(self, obs):
        return int(self.action_best_batch([obs])[0])

    def update(self, batch, grad_step: int):
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(batch["act"], dtype=torch.long, device=self.device).view(-1)
        rew_t = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)

        loss = self._loss(obs_t, act_t, rew_t, next_obs_t, done_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if grad_step % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    # ====================
    # cql mathmatics
    # ====================
    def _td_target(self, next_obs_t: torch.Tensor, rew_t: torch.Tensor, done_t: torch.Tensor) -> torch.Tensor:
        # target = r + gamma * max Q_target(s', a'_policy) * (1-done)
        with torch.no_grad():
            next_action = self.q_network(next_obs_t).argmax(dim=1)
            next_q = self.target_q_network(next_obs_t).gather(1, next_action.view(-1, 1))
            return rew_t + self.gamma * next_q * (1.0 - done_t)

    def _loss_td(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_t: torch.Tensor,
        next_obs_t: torch.Tensor,
        done_t: torch.Tensor,
    ) -> torch.Tensor:
        # L_TD = E[ (Q(s,a) - target)^2 ]
        q_values = self.q_network(obs_t)
        chosen_q = q_values.gather(1, act_t.view(-1, 1)) # Q(si,ai) for batch
        target = self._td_target(next_obs_t, rew_t, done_t)

        # Huber loss
        diff = target - chosen_q
        cond = diff.detach().abs() < 1.0
        huber = torch.where(cond, 0.5 * diff**2, diff.abs() - 0.5)
        return huber.mean()

    def _loss_cql(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        # L_CQL(H) = E_s~D[logsumexp(Q)] - E_(s,a)~D[Q]
        values = self.q_network(obs_t)
        logsumexp = torch.logsumexp(values, dim=1, keepdim=True)

        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size).float()
        data_values = (values * one_hot).sum(dim=1, keepdim=True)

        return (logsumexp - data_values).mean()

    def _loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_t: torch.Tensor,
        next_obs_t: torch.Tensor,
        done_t: torch.Tensor,
    ) -> torch.Tensor:
        # L = L_TD + alpha * L_CQL(H)
        loss_td = self._loss_td(obs_t, act_t, rew_t, next_obs_t, done_t)
        loss_cql = self._loss_cql(obs_t, act_t)
        return loss_td + self.alpha * loss_cql
