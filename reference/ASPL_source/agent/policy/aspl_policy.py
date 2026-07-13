import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from .base_policy import TD3Policy
from scipy.stats import qmc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ASPLPolicy(TD3Policy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        num_sampled_actions: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        learning_rate: float = 3e-4,
        use_lr_scheduler: str = 'critic',
        max_timesteps: int = 1000000,
    ) -> None:
        super().__init__(
            state_dim, action_dim, max_action, discount, tau, 
            policy_noise, noise_clip, policy_freq, hidden_dim, 
            num_layers, learning_rate
        )
        
        self.alpha = alpha
        self.num_sampled_actions = num_sampled_actions
        self.use_lr_scheduler = use_lr_scheduler
        
        # Initialize mean_abs_q for global factored mse
        self.mean_abs_q = 0.0
        self.total_it = 0
        
        # Initialize learning rate schedulers if enabled
        self.actor_scheduler = None
        self.critic_scheduler = None
        if use_lr_scheduler in ['actor', 'both']:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer, T_max=max_timesteps, eta_min=learning_rate * 0.01
            )
        if use_lr_scheduler in ['critic', 'both']:
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer, T_max=max_timesteps, eta_min=learning_rate * 0.01
            )

    def sample_actions(
        self,
        batch_size: int,
        action_dim: int,
    ) -> torch.Tensor:
        """Sample actions for unsupervised learning
        
        Args:
            batch_size: Size of the batch
            action_dim: Dimension of actions
        
        Returns:
            Sampled actions with shape (num_uniform_actions, batch_size, action_dim)
        """
        # Fixed to Latin Hypercube Sampling (LHS)
        sampler = qmc.LatinHypercube(d=action_dim)
        samples = sampler.random(n=self.num_sampled_actions)
        scaled_samples = qmc.scale(samples, [-self.max_action] * action_dim, [self.max_action] * action_dim)
        sampled_actions_base = torch.FloatTensor(scaled_samples).to(device)
        sampled_actions = sampled_actions_base.unsqueeze(1).repeat(1, batch_size, 1)
        
        return sampled_actions

    def update(self, replay_buffer: Any, batch_size: int = 256) -> Dict[str, Optional[float]]:
        self.total_it += 1

        # ========== Step 1: Sample Data and Prepare Inputs ==========
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # ========== Step 2: Compute Critic Loss ==========
        # --------- Compute Supervised Critic Loss ---------
        # Compute target Q-values using parent method
        target_Q = self._compute_target_q(next_state, reward, not_done)

        # Get current Q estimates
        action_Q1, action_Q2 = self.critic(state, action)

        # Compute supervised critic loss
        critic_loss_supervised = F.mse_loss(action_Q1, target_Q) + F.mse_loss(action_Q2, target_Q)

        # --------- Compute Unsupervised Critic Loss ---------
        # Sampling is fixed to LHS
        sampled_actions = self.sample_actions(
            batch_size,
            action.shape[1],
        )

        # (batch_size, action_dim) -> (num_uniform_actions, batch_size, action_dim)
        action_expanded = action.unsqueeze(0).expand(self.num_sampled_actions, -1, -1)
        action_diff = (sampled_actions - action_expanded) ** 2
        f_penalty = action_diff / (2 * self.max_action) ** 2  # (num_uniform_actions, batch_size, action_dim)
        f_penalty_avg = f_penalty.mean(dim=2).view(-1, 1)  # (num_uniform_actions * batch_size, 1)
        
        # (num_uniform_actions, batch_size, action_dim) -> (num_uniform_actions * batch_size, action_dim)
        sampled_actions_flat = sampled_actions.view(-1, action.shape[1])
        # (batch_size, state_dim) -> (num_uniform_actions, batch_size, state_dim) -> (num_uniform_actions * batch_size, state_dim)
        state_expanded_flat = state.unsqueeze(0).expand(self.num_sampled_actions, -1, -1).reshape(-1, state.shape[1])

        # calculate Q-values for uniform actions
        # (num_uniform_actions * batch_size, 1)
        Q1_sampled, Q2_sampled = self.critic(state_expanded_flat, sampled_actions_flat)
        # (batch_size, 1) -> (num_uniform_actions * batch_size, 1)
        target_Q_expanded = target_Q.repeat(self.num_sampled_actions, 1)
        
        # SPL loss
        # (num_uniform_actions * batch_size, 1)
        pseudo_labels_Q1, pseudo_labels_Q2 = self._compute_pseudo_labels(target_Q_expanded, f_penalty_avg)
        critic_loss_unsupervised = F.mse_loss(Q1_sampled, pseudo_labels_Q1) + \
                                  F.mse_loss(Q2_sampled, pseudo_labels_Q2)

        # ========== Step 3: Optimize Critic Networks ==========
        # Total critic loss
        critic_loss = critic_loss_supervised + self.alpha * critic_loss_unsupervised

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update learning rate scheduler if enabled
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # ========== Step 4: Store Losses ==========
        # Store losses for logging
        losses = {
            'critic_loss': critic_loss.item(),
            'critic_loss_supervised': critic_loss_supervised.item(),
            'critic_loss_unsupervised': critic_loss_unsupervised.item(),
            'actor_loss': None
        }

        # ========== Step 5: Delayed Policy Updates ==========
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss using parent method
            actor_loss = self._compute_actor_loss(state)
            losses['actor_loss'] = actor_loss.item()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update learning rate scheduler if enabled
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()

            # Update the frozen target models using parent method
            self._update_target_networks()
            
        return losses 

    def _compute_pseudo_labels(self, target_Q_expanded, f_penalty_avg):
        with torch.no_grad():
            # Global factored MSE (only supported method)
            current_batch_mean_abs_q = torch.abs(target_Q_expanded).mean().item()
            self.mean_abs_q = ((self.total_it - 1) * self.mean_abs_q + current_batch_mean_abs_q) / self.total_it
            
            pseudo_labels_Q1 = target_Q_expanded - f_penalty_avg * self.mean_abs_q
            pseudo_labels_Q2 = target_Q_expanded - f_penalty_avg * self.mean_abs_q
        
        return pseudo_labels_Q1, pseudo_labels_Q2 


