import numpy as np
import torch
import torch.nn.functional as F
from ice_offline.agent._spec import Agent, MetricValues
from ice_offline.dataset._types import Batch


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(o))


class _Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.pi = _Pi(obs_size, act_size)


class BCDeterministicAgent(Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda"):
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.actor = _Actor(self.obs_size, self.act_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

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
            else:
                return super().eval(observations, actions, method)
        return values.cpu().numpy().astype(np.float32)

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch) -> None:
        self.update_actor(batch)

    def update_actor(self, batch: Batch) -> None:
        self.actor_optimizer.zero_grad()
        loss = self.loss_actor(batch)
        loss.backward()
        self.actor_optimizer.step()

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        self.actor_optimizer.zero_grad()
        loss_actor = self.loss_actor(batch)
        grad_actor = self._grad_norm(loss_actor, self.actor.parameters())
        loss_actor.backward()
        self.actor_optimizer.step()

        return {
            "loss_actor": loss_actor.detach(),
            "grad_actor": grad_actor.detach(),
        }

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])

    # ====================
    # Actor loss
    # ====================
    def loss_actor(self, batch: Batch) -> torch.Tensor:
        o, a, _, _, _ = batch
        a_pred = self.actor.pi(o)
        return F.mse_loss(a_pred, a)


