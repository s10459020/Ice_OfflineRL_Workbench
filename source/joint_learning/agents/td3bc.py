from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from joint_learning.datasets.lib import Batch


class Policy(torch.nn.Module):
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


class TD3BCAgent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        discount_factor: float = 0.99,
        target_update_rate: float = 0.005,
        update_actor_interval: int = 2,
        weight_td3: float = 2.5,
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
        self.weight_td3 = weight_td3
        self.max_action = max_action
        self.noise_scale = noise_scale * max_action
        self.noise_clip = noise_clip

        self.policy = Policy(obs_size, act_size, self.max_action).to(device)
        self.target_policy = Policy(obs_size, act_size, self.max_action).to(device)
        self.q1 = QNetwork(obs_size, act_size).to(device)
        self.q2 = QNetwork(obs_size, act_size).to(device)
        self.target_q1 = QNetwork(obs_size, act_size).to(device)
        self.target_q2 = QNetwork(obs_size, act_size).to(device)
        self.sync_target_hard()

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters())
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()))

    # ====================
    # Public Function
    # ====================
    def act(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observations = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.policy(observations)
        return action.cpu().numpy()[0]

    def update(self, batch: Batch) -> None:
        self.update_step += 1
        loss_critic = self.loss_td(batch)
        self.q_optimizer.zero_grad()
        loss_critic.backward()
        self.q_optimizer.step()

        if self.update_step % self.update_actor_interval == 0:
            loss_actor = self.loss_actor(batch)
            self.policy_optimizer.zero_grad()
            loss_actor.backward()
            self.policy_optimizer.step()
            self.sync_target_soft()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)

    # ====================
    # Help Function
    # ====================
    def sync_target_hard(self) -> None:
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def sync_target_soft(self) -> None:
        with torch.no_grad():
            for source, target in zip(self.policy.parameters(), self.target_policy.parameters()):
                target.data.copy_(self.target_update_rate * source.data + (1.0 - self.target_update_rate) * target.data)
            for source, target in zip(self.q1.parameters(), self.target_q1.parameters()):
                target.data.copy_(self.target_update_rate * source.data + (1.0 - self.target_update_rate) * target.data)
            for source, target in zip(self.q2.parameters(), self.target_q2.parameters()):
                target.data.copy_(self.target_update_rate * source.data + (1.0 - self.target_update_rate) * target.data)

    # ====================
    # Loss Function
    # ====================
    def target_td3(self, next_observations: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        # TD3 target:
        #   a' = clip(pi_target(s') + epsilon, -a_max, a_max)
        #   y = r + gamma * (1 - d) * min_i Q_i_target(s', a')
        # Target policy smoothing makes the critic less sensitive to sharp action errors.
        with torch.no_grad():
            next_actions = self.target_policy(next_observations)
            noise = torch.randn_like(next_actions) * self.noise_scale
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            target_q = torch.minimum(
                self.target_q1(next_observations, next_actions),
                self.target_q2(next_observations, next_actions),
            )
            return rewards + self.discount_factor * target_q * (1.0 - dones)

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # Critic TD objective:
        #   L_Q = E_D[ (Q_1(s,a) - y)^2 + (Q_2(s,a) - y)^2 ]
        # The target y is computed with clipped double Q-learning.
        observations, actions, rewards, next_observations, dones = batch
        target = self.target_td3(next_observations, rewards, dones)
        q1 = self.q1(observations, actions)
        q2 = self.q2(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def loss_normal(self, batch: Batch) -> torch.Tensor:
        observations, _, _, _, _ = batch
        policy_actions = self.policy(observations)
        q = torch.minimum(
            self.q1(observations, policy_actions),
            self.q2(observations, policy_actions),
        )
        return -q.mean() / q.abs().mean().detach()

    def loss_bc(self, batch: Batch) -> torch.Tensor:
        # Behavior cloning regularizer:
        #   L_BC = E_{(s,a)~D}[ || pi(s) - a ||_2^2 ]
        # This keeps the learned policy close to dataset actions in offline RL.
        observations, actions, _, _, _ = batch
        predicted_actions = self.policy(observations)
        return F.mse_loss(predicted_actions, actions)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        loss_normal = self.loss_normal(batch)
        loss_bc = self.loss_bc(batch)
        return self.weight_td3 * loss_normal + loss_bc
