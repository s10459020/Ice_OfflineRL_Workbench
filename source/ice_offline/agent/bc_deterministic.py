from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from ice_offline.agent._spec import TorchAgent
from ice_offline.dataset._spec import TorchBuffer


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

    def dist(self, o: torch.Tensor) -> torch.Tensor:
        return self.network(o)

    def mode(self, o: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dist(o))

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        return self.mode(o)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.mode(o)


@dataclass
class BCAgentDeterministic(TorchAgent):
    obs_size: int
    act_size: int
    learning_rate: float = 1e-3
    device: str = "cpu"

    # ====================
    # Init
    # ====================
    def __post_init__(self):
        self.policy = _Pi(self.obs_size, self.act_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Public API
    # ====================
    def act(self, observation):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.policy.mode(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.policy.mode(o)
        return a.cpu().numpy()

    def update(self, batch: TorchBuffer):
        loss = self.loss_actor(batch.obs_list, batch.act_list)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pi": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state["pi"])
        self.optimizer.load_state_dict(state["optimizer"])

    # ====================
    # bc mathmatics
    # ====================
    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # loss_BC = E{log pi(a|s)}
        # if pi is gaussian, loss_BC => MSE(a, pi(s))
        a_pred = self.policy.mode(o)
        return F.mse_loss(a_pred, a)
