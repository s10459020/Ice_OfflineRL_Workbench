import torch
import torch.nn.functional as F

from ice_offline.agent._spec import Agent
from ice_offline.agent._spec import MetricValues
from ice_offline.dataset._types import Batch


class _M(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a], -1)
        return self.network(x)


class NormalizationDynamic(Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.noise_scale = config.get("noise_scale", 3e-3)
        self.state_mean = torch.as_tensor(config["state_mean"], dtype=torch.float32, device=self.device).reshape(1, self.obs_size)
        self.state_std = torch.as_tensor(config["state_std"], dtype=torch.float32, device=self.device).reshape(1, self.obs_size)
        self.model = _M(self.obs_size, self.act_size, config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    # ====================
    # extend
    # ====================
    def prepare(self) -> "NormalizationDynamic":
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        return self

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pred = self.model(self.normalize_state(o), a)
        return self.denormalize_state(pred)

    def noise_state(self, o: torch.Tensor) -> torch.Tensor:
        o_normal = self.normalize_state(o)
        noise = torch.randn(o_normal.shape, device=o.device) * self.noise_scale
        return self.denormalize_state(o_normal + noise)

    def normalize_state(self, o: torch.Tensor) -> torch.Tensor:
        return (o - self.state_mean) / self.state_std

    def denormalize_state(self, o: torch.Tensor) -> torch.Tensor:
        return o * self.state_std + self.state_mean

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_dynamic",
            "grad_dynamic",
        ]

    def update(self, batch: Batch) -> MetricValues:
        loss_dynamic, metrics = self.loss_dynamic(batch)
        self.optimizer.zero_grad()
        loss_dynamic.backward()
        self.optimizer.step()
        return metrics

    # ====================
    # extend
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "model": self.model.state_dict(),
            "state_mean": self.state_mean,
            "state_std": self.state_std,
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state["model"])
        self.state_mean = state["state_mean"].to(self.device)
        self.state_std = state["state_std"].to(self.device)

    # ====================
    # mathmatics
    # ====================
    def loss_dynamic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s, a, _, sn, _ = batch
        pred = self.model(self.normalize_state(s), a)
        loss = F.mse_loss(pred, self.normalize_state(sn))
        return loss, {
            "loss_dynamic": self._value(loss.detach()),
            "grad_dynamic": self._grad_norm(loss, self.model.parameters()),
        }
