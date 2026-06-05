from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# (o, a, r, o', d), with r and d shaped as column tensors.
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

@dataclass
class Episode:
    observations: np.ndarray | dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    infos: dict[str, Any] | None


@dataclass
class Buffer:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass
class Metadata:
    env_id: str = ""
    obs_shape: tuple[int, ...] = ()
    act_shape: tuple[int, ...] = ()
    obs_dim: int = 0
    act_dim: int = 0
    count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
