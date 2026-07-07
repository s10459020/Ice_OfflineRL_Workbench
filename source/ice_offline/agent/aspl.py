import torch
import torch.nn.functional as F
from scipy.stats import qmc

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.td3 import TD3Actor
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3 import TD3Critic
from ice_offline.dataset._types import Batch


class AsplActor(TD3Actor):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__(obs_size, act_size, config)
        self.num_sample = config.get("actor_num_sample", 5)
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_size, seed=config.get("actor_seed", 42))

    def set_seed(self, seed: int) -> None:
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_size, seed=seed)

    def sample_actions_lhs(self, batch_size: int) -> torch.Tensor:
        samples = self._lhs_sampler.random(n=self.num_sample) # (N, A)[0 ~ 1]

        samples = qmc.scale(
            samples, 
            [-self.max_action] * self.act_size, 
            [self.max_action] * self.act_size
        ) # [-a ~ a]

        samples = torch.as_tensor(
            samples,
            dtype=torch.float32,
            device=next(self.pi.parameters()).device
        )

        samples = samples.unsqueeze(1) # (N, 1, A)
        samples = samples.repeat(1, batch_size, 1) # (N, B, A)
        return samples

    def action_distance(self, a: torch.Tensor, a_samples: torch.Tensor) -> torch.Tensor:
        # d(a,a~) = [ (a - a~) / (a_max - a_min) ]^2
        # source use constant (a_max-a_min) ~= 2*a_max
        # d range: [-0.5, 0.5]
        # a shape: (B, A)
        # a_samples shape: (N, B, A)
        diff = (a - a_samples)**2 # (N, B, A) 
        normalize = (2 * self.max_action)**2
        result = diff / normalize 
        return result.mean(dim=2, keepdim=True) # (N, B, 1)


class AsplCritic(TD3Critic):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__(obs_size, act_size, config)
        self.rate_decay = config.get("critic_rate_decay", 0.005)
        self.q_avg = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)

    def update_q_avg(self, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current = target.abs().mean()
            if self.q_avg.item() == 0.0:
                self.q_avg.copy_(current)
            else:
                self.q_avg.mul_(1.0 - self.rate_decay)
                self.q_avg.add_(self.rate_decay * current)
        return self.q_avg

    def q_pseudo(self, q_target: torch.Tensor, action_distance: torch.Tensor) -> torch.Tensor:
        # Q~(s,a~) = Q^(s,a) - c * d(a,a~)
        # c is scaling coefficent
        # q use q_target in source
        # q_target range: [-Q, Q], shape: (B, 1)
        with torch.no_grad():
            return q_target - self.q_avg * action_distance  # (N, B, 1)


class AsplAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.weight_punish = config.get("weight_punish", 0.05)
        super().__init__(obs_size=obs_size, act_size=act_size, config=config, device=device)
        self.actor = AsplActor(self.obs_size, self.act_size, config).to(self.device)
        self.critic = AsplCritic(self.obs_size, self.act_size, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.param_actor())
        self.critic_optimizer = torch.optim.Adam(self.critic.param_critic())

    def set_seed(self, seed: int) -> None:
        self.actor.set_seed(seed)

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_punish",
            "grad_punish",
            "loss_critic",
            "grad_critic",
            "loss_td3",
            "grad_td3",
            "q_avg",
            "target_q",
        ]

    def update(self, batch: Batch) -> MetricValues:
        _, _, r, sn, d = batch
        target = self.target_td3(sn, r, d)
        self.update_step += 1

        q_avg = self.critic.update_q_avg(target)
        metrics = self.update_critic(batch)
        metrics["q_avg"] = self._value(q_avg.detach())

        if self.update_step % self.update_actor_interval == 0:
            metrics |= self.update_actor(batch)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

        return metrics

    # ====================
    # Critic loss
    # ====================    
    def loss_punish(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s, a, r, sn, d = batch
        target = self.target_td3(sn, r, d)

        # E_{s~D}{(a~)~U}[ Q(s,a~) - Q~(s,a~) ]^2
        a_samples = self.actor.sample_actions_lhs(s.shape[0])       # (N,B,A)
        action_distance = self.actor.action_distance(a, a_samples)  # (N,B,1)
        q_pseudo = self.critic.q_pseudo(target, action_distance)    # (N,B,1)

        # reshape
        s_reshape = s.unsqueeze(0).expand(a_samples.shape[0], -1, -1).reshape(-1, s.shape[1])    # (B,S) > (1,B,S) > (N,B,S) > (N*B,S)
        a_samples_reshape = a_samples.view(-1, a.shape[1])                                       # (N,B,A) > (N*B,A)
        q_pseudo_reshape = q_pseudo.view(-1, 1)                                                  # (N*B,1)  
        
        q_values = (
            self.critic.q_networks[0](s_reshape, a_samples_reshape),
            self.critic.q_networks[1](s_reshape, a_samples_reshape),
        )
        losses = [F.mse_loss(q_value, q_pseudo_reshape) for q_value in q_values]
        loss = sum(losses)
        return loss, {
            "loss_punish": self._value(loss.detach()),
            "grad_punish": self._grad_norm(loss, self.critic.param_critic()),
        }

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # loss = TD + weight_punish * Punish
        loss_td, metrics_td = self.loss_td(batch)
        loss_punish, metrics_punish = self.loss_punish(batch)
        loss = loss_td + self.weight_punish * loss_punish
        return loss, metrics_td | metrics_punish | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }
