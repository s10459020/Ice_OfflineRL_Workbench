import numpy as np
import torch
import torch.nn.functional as F

from ice_offline.agent._spec import Agent
from ice_offline.agent._spec import MetricValues
from ice_offline.dataset._types import Batch


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.max_action = config.get("max_action", 1.0)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(o))


class _Q(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([o, a], dim=1))


class TD3Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, pi_cls: type[torch.nn.Module] = _Pi):
        super().__init__()
        self.obs_size = obs_size
        self.act_size = act_size
        self.max_action = config.get("max_action", 1.0)
        self.target_update_rate = config.get("target_update_rate", 0.005)
        self.noise_scale = config.get("noise_scale", 0.2) * self.max_action
        self.noise_clip = config.get("noise_clip", 0.5)
        self.pi = pi_cls(obs_size, act_size, config)
        self.tpi = pi_cls(obs_size, act_size, config)
        self.sync_target_hard()

    def noise_action(self, a: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(a) * self.noise_scale
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        return (a + noise).clamp(-self.max_action, self.max_action)

    def sample_random_n(self, o: torch.Tensor, count: int) -> torch.Tensor:
        batch_size = o.shape[0]
        return torch.empty(
            (batch_size, count, self.act_size),
            device=o.device,
            dtype=o.dtype,
        ).uniform_(-self.max_action, self.max_action)

    # ====================
    # TD3 target sync
    # ====================
    def sync_target_hard(self) -> None:
        self.tpi.load_state_dict(self.pi.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.pi.parameters(), self.tpi.parameters()):
                tp.data.copy_(
                    self.target_update_rate * p.data
                    + (1.0 - self.target_update_rate) * tp.data
                )

    # ====================
    # Params
    # ====================
    def param_actor(self):
        return self.pi.parameters()


class TD3Critic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, q_count: int = 2, q_cls: type[torch.nn.Module] = _Q):
        super().__init__()
        self.target_update_rate = config.get("target_update_rate", 0.005)
        self.q_networks = torch.nn.ModuleList(
            [q_cls(obs_size, act_size) for _ in range(q_count)]
        )
        self.tq_networks = torch.nn.ModuleList(
            [q_cls(obs_size, act_size) for _ in range(q_count)]
        )
        self.sync_target_hard()

    # ====================
    # Public API
    # ====================
    def q_all(self, o: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(q(o, a) for q in self.q_networks)

    def tq_all(self, o: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(tq(o, a) for tq in self.tq_networks)

    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(o, a), dim=1).min(dim=1, keepdim=True).values

    def q_mean(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(o, a), dim=1).mean(dim=1, keepdim=True)

    def tq_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.tq_all(o, a), dim=1).min(dim=1, keepdim=True).values

    def tq_mean(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.tq_all(o, a), dim=1).mean(dim=1, keepdim=True)

    # ====================
    # TD3 target sync
    # ====================
    def sync_target_hard(self) -> None:
        for q, tq in zip(self.q_networks, self.tq_networks):
            tq.load_state_dict(q.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for q, tq in zip(self.q_networks, self.tq_networks):
                for p, tp in zip(q.parameters(), tq.parameters()):
                    tp.data.copy_(
                        self.target_update_rate * p.data
                        + (1.0 - self.target_update_rate) * tp.data
                    )

    # ====================
    # Params
    # ====================
    def param_critic(self):
        return self.q_networks.parameters()


class TD3Agent(Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = config.get("discount_factor", 0.99)
        self.update_actor_interval = config.get("update_actor_interval", 2)
        self.update_step = 0
        self.actor = TD3Actor(self.obs_size, self.act_size, config=config).to(self.device)
        self.critic = TD3Critic(
            self.obs_size,
            self.act_size,
            config=config,
            q_count=int(config.get("q_count", 2)),
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.param_actor())
        self.critic_optimizer = torch.optim.Adam(self.critic.param_critic())

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(int(seed))

    # ====================
    # Act
    # ====================
    def act(self, observation):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(o)
        return a.cpu().numpy()

    def eval(self, observations, actions, method: str) -> np.ndarray:
        o = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if method == "Pi":
                mode = self.actor.pi(o)
                sigma = 0.25
                diff = ((mode - a) ** 2).mean(dim=-1)
                values = torch.exp(-0.5 * diff / (sigma * sigma))
            elif method == "Q":
                values = self.critic.q_min(o, a).squeeze(-1)
            else:
                return super().eval(observations, actions, method)
        return values.cpu().numpy().astype(np.float32)

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_td",
            "grad_td",
            "loss_td3",
            "grad_td3",
            "param_q",
            "target_q",
        ]

    def update(self, batch: Batch) -> MetricValues:
        self.update_step += 1
        metrics = self.update_critic(batch)
        if self.update_step % self.update_actor_interval == 0:
            metrics |= self.update_actor(batch)
            self.critic.update_target_soft()
            self.actor.update_target_soft()
        return metrics

    def update_critic(self, batch: Batch) -> MetricValues:
        loss_critic, metrics = self.loss_critic(batch)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        return metrics

    def update_actor(self, batch: Batch) -> MetricValues:
        loss_actor, metrics = self.loss_actor(batch)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        return metrics

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def _load_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_step = int(state["update_step"])

    # ====================
    # Critic loss
    # ====================
    def target_td3(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # y = r + discount_factor * Q(s',pi(s'))
        with torch.no_grad():
            an = self.actor.tpi(on)
            an = self.actor.noise_action(an)
            tq = self.critic.tq_min(on, an)
            return r + self.discount_factor * tq * (1.0 - d)

    def loss_td(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # loss = E{s,a,r,s'~D}[ MSE(Q(s,a) - y) ]
        o, a, r, on, d = batch
        target = self.target_td3(on, r, d)
        loss = sum(F.mse_loss(q, target) for q in self.critic.q_all(o, a))
        
        param_q = torch.zeros((), device=loss.device)
        for param in self.critic.param_critic():
            param_q = param_q + param.detach().square().sum()

        return loss, {
            "loss_td": self._value(loss.detach()),
            "grad_td": self._grad_norm(loss, self.critic.param_critic()),
            "param_q": self._value(param_q.sqrt()),
            "target_q": self._value(target.mean().detach()),
        }

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss_td(batch)

    # ====================
    # Actor loss
    # ====================
    def loss_td3(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # loss = E{s~D}[ -Q(s,pi(s)) ]
        o, _, _, _, _ = batch
        a = self.actor.pi(o)
        q = self.critic.q_min(o, a)
        loss = -q.mean()
        return loss, {
            "loss_td3": self._value(loss.detach()),
            "grad_td3": self._grad_norm(loss, self.actor.param_actor()),
        }

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss_td3(batch)
