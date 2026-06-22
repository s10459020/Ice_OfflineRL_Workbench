import torch
import torch.nn.functional as F
from scipy.stats import qmc

from ice_offline.agent._spec import MetricValues
from ice_offline.agent.td3 import TD3Actor
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3 import TD3Critic
from ice_offline.dataset._types import Batch


class AsplActor(TD3Actor):
    def __init__(self, seed: int = 42, num_sample: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_sample = num_sample
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_size, seed=seed)

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
    def __init__(self, rate_decay: float = 0.005, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_decay = rate_decay
        self.scale_punish = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)

    def update_scale_punish(self, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current = target.abs().mean()
            if self.scale_punish.item() == 0.0:
                self.scale_punish.copy_(current)
            else:
                self.scale_punish.mul_(1.0 - self.rate_decay)
                self.scale_punish.add_(self.rate_decay * current)
        return self.scale_punish

    def q_pseudo(self, q_target: torch.Tensor, action_distance: torch.Tensor) -> torch.Tensor:
        # Q~(s,a~) = Q^(s,a) - c * d(a,a~)
        # c is scaling coefficent
        # q use q_target in source
        # q_target range: [-Q, Q], shape: (B, 1)
        with torch.no_grad():
            return q_target - self.scale_punish * action_distance  # (N, B, 1)


class AsplAgent(TD3Agent):
    id: str = "aspl"
    weight_punish: float = 0.5
    learning_rate: float = 3e-4

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        config: dict[str, object] = {},
        device: str = "cuda",
    ) -> None:
        cfg = config
        self.id = str(cfg.get("id", "aspl"))
        self.weight_punish = cfg.get("weight_punish", 0.5)
        self.learning_rate = cfg.get("learning_rate", 3e-4)
        super().__init__(
            obs_size=obs_size,
            act_size=act_size,
            config=cfg,
            device=device,
        )
        self.actor = AsplActor(
            obs_size=self.obs_size,
            act_size=self.act_size,
            seed=cfg.get("actor_seed", 42),
            num_sample=cfg.get("actor_num_sample", 5),
        ).to(self.device)
        self.critic = AsplCritic(
            obs_size=self.obs_size,
            act_size=self.act_size,
            q_count=self.q_count,
            rate_decay=cfg.get("critic_rate_decay", 0.005),
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.pi.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.q_networks.parameters(), lr=self.learning_rate)

    def set_seed(self, seed: int) -> None:
        self.actor.set_seed(seed)

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch) -> None:
        _, _, r, sn, d = batch
        target = self.target_td3(sn, r, d)
        self.update_step += 1
        
        self.critic.update_scale_punish(target)
        
        self.critic_optimizer.zero_grad()
        loss_critic = self.loss_critic(batch, target)
        loss_critic.backward()
        self.critic_optimizer.step()
    
        if self.update_step % self.update_actor_interval == 0:
            self.actor_optimizer.zero_grad()
            loss_actor = self.loss_actor(batch)
            loss_actor.backward()
            self.actor_optimizer.step()

            self.critic.update_target_soft()
            self.actor.update_target_soft()

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        _, _, r, sn, d = batch
        target = self.target_td3(sn, r, d)
        self.update_step += 1

        scale_punish = self.critic.update_scale_punish(target)

        loss_td = self.loss_td(batch)
        grad_td = self._grad_norm(loss_td, self.critic.parameters())

        loss_punish = self.loss_punish(batch)
        grad_punish = self._grad_norm(loss_punish, self.critic.parameters())

        loss_critic = loss_td + self.weight_punish * loss_punish
        grad_critic = self._grad_norm(loss_critic, self.critic.parameters())

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        metrics = {
            "loss_td": loss_td.detach(),
            "grad_td": grad_td.detach(),
            "loss_punish": loss_punish.detach(),
            "grad_punish": grad_punish.detach(),
            "loss_critic": loss_critic.detach(),
            "grad_critic": grad_critic.detach(),
            "loss_actor": None,
            "grad_actor": None,
            "scale_punish": scale_punish.detach(),
            "target_q": target.abs().mean(),
        }

        if self.update_step % self.update_actor_interval == 0:
            loss_actor = self.loss_actor(batch)
            grad_actor = self._grad_norm(loss_actor, self.actor.parameters())

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            self.critic.update_target_soft()
            self.actor.update_target_soft()

            metrics.update({
                "loss_actor": loss_actor.detach(),
                "grad_actor": grad_actor.detach(),
            })

        return metrics

    # ====================
    # Critic loss
    # ====================    
    def loss_td_huber(self, batch: Batch) -> torch.Tensor:
        # loss = E{s,a,r,s'~D}[ MSE(Q(s,a) - y) ]
        o, a, r, on, d = batch
        target = self.target_td3(on, r, d)
        return sum(F.huber_loss(q, target, delta=1.0) for q in self.critic.q_all(o, a))

    def loss_punish(self, batch: Batch) -> torch.Tensor:
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
        return sum(losses)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # loss = TD + weight_punish * Punish
        loss_td = self.loss_td(batch)
        loss_aspl = self.loss_punish(batch)
        return loss_td + self.weight_punish * loss_aspl


    # ====================
    # Actor loss
    # ====================
    def loss_td3(self, batch: Batch) -> torch.Tensor:
        # loss = E{s~D}[ -Q(s,pi(s)) ]
        o, _, _, _, _ = batch
        a = self.actor.pi(o)
        q = self.critic.q_networks[0](o, a)
        return -q.mean()


