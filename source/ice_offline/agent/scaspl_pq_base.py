import copy

import torch
import torch.nn.functional as F

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.scaspl_n import ScasplNAgent
from ice_offline.dataset._types import Batch


class ScasplPQBaseAgent(ScasplNAgent):
    normal_q_source = "value"
    correction_q_source = "value"

    def __init__(self, obs_size: int, act_size: int, dynamics, config: dict[str, object] = {}, device: str = "cuda") -> None:
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            dynamics=dynamics,
            config=config,
            device=device,
        )
        self.critic_punish = copy.deepcopy(self.critic).to(self.device)
        self.critic_punish_optimizer = torch.optim.Adam(self.critic_punish.param_critic())
        self.weight_pi = config.get("weight_pi", 1.0 - self.weight_correction)

    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_pq_td",
            "grad_pq_td",
            "loss_punish",
            "grad_punish",
            "loss_critic",
            "grad_critic",
            "loss_pq_critic",
            "grad_pq_critic",
            "loss_normal",
            "grad_normal",
            "loss_correction",
            "grad_correction",
            "loss_actor",
            "grad_actor",
            "q_avg",
            "q_pq_avg",
            "target_q",
            "target_pq",
        ]

    def update(self, batch: Batch) -> MetricValues:
        metrics = self.update_critic(batch)

        self.update_step += 1
        if self.update_step % self.update_actor_interval == 0:
            metrics |= self.update_actor(batch)
            self.critic.update_target_soft()
            self.critic_punish.update_target_soft()
            self.actor.update_target_soft()

        return metrics

    def update_critic(self, batch: Batch) -> MetricValues:
        _, _, reward, next_observation, terminal = batch
        target = self.target_td3(next_observation, reward, terminal)
        target_punish = self.target_punish_td3(next_observation, reward, terminal)

        loss_td, metrics_td = self.loss_td_clean(batch, target)
        self.critic_optimizer.zero_grad()
        loss_td.backward()
        self.critic_optimizer.step()

        loss_pq_td, metrics_pq_td = self.loss_td_punish(batch, target_punish)
        loss_punish, metrics_punish = self.loss_punish_pq(batch)
        loss_pq_critic = loss_pq_td + self.weight_punish * loss_punish
        metrics_pq_critic = {
            "loss_pq_critic": self._value(loss_pq_critic.detach()),
            "grad_pq_critic": self._grad_norm(loss_pq_critic, self.critic_punish.param_critic()),
        }
        self.critic_punish_optimizer.zero_grad()
        loss_pq_critic.backward()
        self.critic_punish_optimizer.step()

        return metrics_td | metrics_pq_td | metrics_punish | metrics_pq_critic | {
            "q_avg": self._value(self.critic.q_avg.detach()),
            "q_pq_avg": self._value(self.critic_punish.q_avg.detach()),
        }

    def target_punish_td3(self, next_observation: torch.Tensor, reward: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_action = self.actor.tpi(next_observation)
            next_action = self.actor.noise_action(next_action)
            target_q = self.critic_punish.tq_min(next_observation, next_action)
            return reward + self.discount_factor * target_q * (1.0 - terminal)

    def loss_td_clean(self, batch: Batch, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observation, action, _, _, _ = batch
        loss = sum(F.mse_loss(q_value, target) for q_value in self.critic.q_all(observation, action))
        return loss, {
            "loss_td": self._value(loss.detach()),
            "grad_td": self._grad_norm(loss, self.critic.param_critic()),
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
            "target_q": self._value(target.mean().detach()),
        }

    def loss_td_punish(self, batch: Batch, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observation, action, _, _, _ = batch
        loss = sum(F.mse_loss(q_value, target) for q_value in self.critic_punish.q_all(observation, action))
        return loss, {
            "loss_pq_td": self._value(loss.detach()),
            "grad_pq_td": self._grad_norm(loss, self.critic_punish.param_critic()),
            "target_pq": self._value(target.mean().detach()),
        }

    def loss_punish_pq(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observation, action, _, _, _ = batch
        action_samples = self.actor.sample_actions_uniform(observation.shape[0])
        action_distance = self.actor.action_distance(action, action_samples)
        q_pseudo = self.critic_punish.q_pseudo(observation, action, action_distance)

        observation_reshape = observation.unsqueeze(0).expand(action_samples.shape[0], -1, -1).reshape(-1, observation.shape[1])
        action_samples_reshape = action_samples.view(-1, action.shape[1])
        q_pseudo_reshape = q_pseudo.view(-1, 1)

        q_values = (
            self.critic_punish.q_networks[0](observation_reshape, action_samples_reshape),
            self.critic_punish.q_networks[1](observation_reshape, action_samples_reshape),
        )
        loss = sum(F.mse_loss(q_value, q_pseudo_reshape) for q_value in q_values)
        return loss, {
            "loss_punish": self._value(loss.detach()),
            "grad_punish": self._grad_norm(loss, self.critic_punish.param_critic()),
        }

    def loss_normal(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observation, _, _, _, _ = batch
        action = self.actor.pi(observation)
        critic = self._actor_critic(self.normal_q_source)
        q_value = critic.q_min(observation, action)
        loss = -q_value.mean() / q_value.abs().mean().detach()
        return loss, {
            "loss_normal": self._value(loss.detach()),
            "grad_normal": self._grad_norm(loss, self.actor.param_actor()),
        }

    def loss_correction(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observation, _, _, next_observation, _ = batch
        critic = self._actor_critic(self.correction_q_source)

        action = self.actor.pi(observation)
        value = critic.q_min(observation, action)
        next_action = self.actor.pi(next_observation)
        next_value = critic.q_min(next_observation, next_action)

        weight = (
            self.scale_gap * (next_value.detach() - value.detach())
        ).exp().clamp(max=self.max_gap)

        perturbed_observation = self.dynamics.noise_state(observation)
        perturbed_action = self.actor.pi(perturbed_observation)
        mse_dynamic = (self.dynamics.forward(perturbed_observation, perturbed_action) - next_observation) ** 2
        loss = (weight * mse_dynamic).mean()
        return loss, {
            "loss_correction": self._value(loss.detach()),
            "grad_correction": self._grad_norm(loss, self.actor.param_actor()),
        }

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_normal, metrics_normal = self.loss_normal(batch)
        loss_correction, metrics_correction = self.loss_correction(batch)
        loss = self.weight_pi * loss_normal + self.weight_correction * loss_correction
        return loss, metrics_normal | metrics_correction | {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.param_actor()),
        }

    def _actor_critic(self, source: str):
        if source == "punish":
            return self.critic_punish
        return self.critic

    def _save_dict(self) -> dict[str, object]:
        state = super()._save_dict()
        state["critic_punish"] = self.critic_punish.state_dict()
        state["critic_punish_optimizer"] = self.critic_punish_optimizer.state_dict()
        return state

    def _load_dict(self, state: dict[str, object]) -> None:
        super()._load_dict(state)
        self.critic_punish.load_state_dict(state["critic_punish"])
        self.critic_punish_optimizer.load_state_dict(state["critic_punish_optimizer"])
