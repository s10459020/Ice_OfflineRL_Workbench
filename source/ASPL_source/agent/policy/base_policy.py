import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasePolicy(object):
    """Base class for all policies"""
    
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
    def select_action(self, state):
        """Select action given a state"""
        raise NotImplementedError
        
    def train(self, replay_buffer, batch_size=256):
        """Train the policy"""
        raise NotImplementedError
        
    def save(self, filename):
        """Save the policy"""
        raise NotImplementedError
        
    def load(self, filename):
        """Load the policy"""
        raise NotImplementedError


class TD3Policy(BasePolicy):
    """TD3 base policy implementation"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        hidden_dim=256,
        num_layers=3,
        learning_rate=3e-4
    ):
        super().__init__(state_dim, action_dim, max_action)
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim, num_layers).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # TD3 hyperparameters
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        """Select action using the actor network"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def _update_target_networks(self):
        """Update target networks using soft updates"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _add_action_noise(self, action):
        """Add clipped noise to action for exploration and robustness"""
        noise = (torch.randn_like(action) * self.policy_noise * self.max_action).clamp(-self.noise_clip, self.noise_clip)
        return (action + noise).clamp(-self.max_action, self.max_action)

    def _compute_target_q(self, next_state, reward, not_done):
        """Compute target Q-values for TD3"""
        with torch.no_grad():
            # Get next action from target actor
            next_action = self.actor_target(next_state)
            
            # Add clipped noise for target policy smoothing
            next_action = self._add_action_noise(next_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            
        return target_Q

    def _compute_critic_loss(self, state, action, target_Q):
        """Compute critic loss"""
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        return critic_loss

    def _compute_actor_loss(self, state):
        """Compute actor loss with noisy action sampling for robustness"""
        action = self.actor(state)
        action = self._add_action_noise(action)
        actor_loss = -self.critic.Q1(state, action).mean()
        return actor_loss

    def update(self, replay_buffer, batch_size=256):
        """Basic TD3 training loop (to be extended by subclasses)"""
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute target Q-values
        target_Q = self._compute_target_q(next_state, reward, not_done)

        # Compute and optimize critic loss
        critic_loss = self._compute_critic_loss(state, action, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Store losses for logging
        losses = {
            'critic_loss': critic_loss.item(),
            'actor_loss': None
        }

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = self._compute_actor_loss(state)
            losses['actor_loss'] = actor_loss.item()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            self._update_target_networks()
            
        return losses

    def save(self, filename):
        """Save policy networks and optimizers"""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        """Load policy networks and optimizers"""
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor) 