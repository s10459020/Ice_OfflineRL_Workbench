from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_learning.lib.dataset import Batch


class TD3Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, max_action: float = 1.0) -> None:
        super().__init__()
        self.max_action = max_action
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(observations))


class QNetwork(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=1))


class TD3Critic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.q_networks = torch.nn.ModuleList(
            [QNetwork(obs_size, act_size), QNetwork(obs_size, act_size)]
        )

    def q_all(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(q_network(observations, actions) for q_network in self.q_networks)

    def q_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(observations, actions), dim=1).min(dim=1, keepdim=True).values

    def q_mean(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(observations, actions), dim=1).mean(dim=1, keepdim=True)

    def action_gradients(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gradients = []
        for q_network in self.q_networks:
            q = q_network(observations, actions)
            gradient = torch.autograd.grad(
                q.sum(),
                actions,
                create_graph=True,
                retain_graph=True,
            )[0]
            gradients.append(gradient)
        return tuple(gradients)


class TD3Agent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        discount_factor: float = 0.99,
        target_update_rate: float = 0.005,
        update_actor_interval: int = 2,
        max_action: float = 1.0,
        noise_scale: float = 0.2,
        noise_clip: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.update_actor_interval = update_actor_interval
        self.update_step = 0
        self.max_action = max_action
        self.noise_scale = noise_scale * max_action
        self.noise_clip = noise_clip

        self.actor = TD3Actor(obs_size, act_size, self.max_action).to(device)
        self.target_actor = TD3Actor(obs_size, act_size, self.max_action).to(device)
        self.critic = TD3Critic(obs_size, act_size).to(device)
        self.target_critic = TD3Critic(obs_size, act_size).to(device)
        self.sync_target_hard()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def act(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observations = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(observations)
        return action.cpu().numpy()[0]

    def update(self, batch: Batch) -> None:
        self.update_step += 1
        loss_critic = self.loss_critic(batch)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        if self.update_step % self.update_actor_interval == 0:
            loss_actor = self.loss_actor(batch)
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            self.sync_target_soft()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state)

    # -------------------------------------------------------------------------
    # Help functions
    # -------------------------------------------------------------------------
    def sync_target_hard(self) -> None:
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def sync_target_soft(self) -> None:
        with torch.no_grad():
            for source, target in zip(self.actor.parameters(), self.target_actor.parameters()):
                target.data.copy_(self.target_update_rate * source.data + (1.0 - self.target_update_rate) * target.data)
            for source, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.data.copy_(self.target_update_rate * source.data + (1.0 - self.target_update_rate) * target.data)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def target_td3(self, next_observations: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        # \epsilon \sim clip(N(0, \sigma^2 I), -c, c)
        # a' = clip(\pi'(s')+\epsilon,-a_(max) ,a_(max) )
        # y = r + \gamma(1-d) min_i Q'_i (s',a')
        with torch.no_grad():
            next_actions = self.target_actor(next_observations)
            noise = (torch.randn_like(next_actions) * self.noise_scale).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            target_q = self.target_critic.q_min(next_observations, next_actions)
            return rewards + self.discount_factor * target_q * (1.0 - dones)

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # Loss_TD = E_D [\sum _(i=1)^2 (Q_i (s,a)-y)^2]
        observations, actions, rewards, next_observations, dones = batch
        target = self.target_td3(next_observations, rewards, dones)
        q1, q2 = self.critic.q_all(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_TD
        return self.loss_td(batch)

    def loss_td3(self, batch: Batch) -> torch.Tensor:
        # Loss_TD3 = -E_D [min_i Q_i (s,\pi(s))]
        observations, _, _, _, _ = batch
        policy_actions = self.actor(observations)
        q = self.critic.q_min(observations, policy_actions)
        return -q.mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = Loss_TD3
        return self.loss_td3(batch)
