from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ice_offline.joint_learning.dataset import Batch


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
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = 0.99
        self.target_update_rate = 0.005
        self.update_actor_interval = 2
        self.update_step = 0
        self.weight_td3 = 0.01
        self.max_action = 1.0
        self.noise_scale = 0.2
        self.noise_clip = 0.5

        self.policy = Policy(obs_size, act_size, self.max_action).to(device)
        self.target_policy = Policy(obs_size, act_size, self.max_action).to(device)
        self.q1 = QNetwork(obs_size, act_size).to(device)
        self.q2 = QNetwork(obs_size, act_size).to(device)
        self.target_q1 = QNetwork(obs_size, act_size).to(device)
        self.target_q2 = QNetwork(obs_size, act_size).to(device)
        self.sync_target_hard()

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters())
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()))

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

    def loss_td3(self, batch: Batch) -> torch.Tensor:
        # Deterministic policy objective:
        #   L_TD3 = E_{s~D}[ -Q_1(s, pi(s)) ]
        # Minimizing this loss increases the critic value of policy actions.
        observations, _, _, _, _ = batch
        policy_actions = self.policy(observations)
        q = self.q1(observations, policy_actions)
        return -q.mean()

    def loss_bc(self, batch: Batch) -> torch.Tensor:
        # Behavior cloning regularizer:
        #   L_BC = E_{(s,a)~D}[ || pi(s) - a ||_2^2 ]
        # This keeps the learned policy close to dataset actions in offline RL.
        observations, actions, _, _, _ = batch
        predicted_actions = self.policy(observations)
        return F.mse_loss(predicted_actions, actions)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # TD3+BC actor objective:
        #   L_pi = alpha * L_TD3 + L_BC
        # The Q term improves actions, while BC constrains them near the dataset support.
        loss_td3 = self.loss_td3(batch)
        loss_bc = self.loss_bc(batch)
        return self.weight_td3 * loss_td3 + loss_bc
