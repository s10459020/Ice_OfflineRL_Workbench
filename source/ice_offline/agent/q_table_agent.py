import pickle
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ._agent_interface import model_path


QTableState = Any
ObservationEncoder = Callable[[Any], QTableState]


class _QTable:
    # ====================
    # Init
    # ====================
    def __init__(self, n_actions: int, encoder: ObservationEncoder) -> None:
        self._n_actions = n_actions
        self._encoder = encoder
        self._table: dict[QTableState, np.ndarray] = {}

    def _ensure(self, key: QTableState) -> np.ndarray:
        q_row = self._table.get(key)
        if q_row is None:
            q_row = np.zeros(self._n_actions, dtype=np.float32)
            self._table[key] = q_row
        return q_row

    # ====================
    # Dunder
    # ====================
    def __getitem__(self, observation: Any) -> np.ndarray:
        key = self._encoder(observation)
        return self._ensure(key)

    def __call__(self, observation: Any, action: int | None = None) -> np.ndarray | float:
        key = self._encoder(observation)
        q_row = self._ensure(key)
        if action is None:
            return q_row
        return q_row[action]

    def __len__(self) -> int:
        return len(self._table)

    # ====================
    # Persistence
    # ====================
    def to_dict(self) -> dict[QTableState, np.ndarray]:
        return self._table

    def load_dict(self, table: dict[QTableState, np.ndarray]) -> None:
        self._table = table


class QTableAgent:
    """Tabular Q-learning agent (action selection + TD update)."""

    # ====================
    # Init
    # ====================
    def __init__(
        self,
        n_actions: int,
        encoder: ObservationEncoder,
        alpha: float = 0.1,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self._rng = random.Random(seed)
        self.Q = _QTable(self.n_actions, encoder)

    # ====================
    # Public API
    # ====================
    def policy(self, o: Any, epsilon: float = 0.0) -> int:
        if self._rng.random() < epsilon:
            return self._rng.randrange(self.n_actions)
        
        q_row = self.Q[o]
        best_value = np.max(q_row)
        best_actions = np.flatnonzero(q_row == best_value)
        
        if len(best_actions) == 1:
            return best_actions[0]
        return self._rng.choice(best_actions.tolist())

    def update(self, o: Any, a: int, r: float, o_: Any, d: bool) -> None:
        q_s = self.Q[o]
        q_next = self.Q[o_]
        td_target = r + (0.0 if d else self.gamma * np.max(q_next))
        q_s[a] += self.alpha * (td_target - q_s[a])

    # ====================
    # Persistence
    # ====================
    def save(self, model_id: str | Path, step: int) -> Path:
        path = model_path(model_id, step, ".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_table": self.Q.to_dict(),
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def _load_from_path(
        cls,
        path: str | Path,
        encoder: ObservationEncoder,
    ) -> "QTableAgent":
        model_path = Path(path)
        with model_path.open("rb") as f:
            payload = pickle.load(f)

        agent = cls(
            n_actions=payload["n_actions"],
            encoder=encoder,
            alpha=payload["alpha"],
            gamma=payload["gamma"],
        )
        agent.Q.load_dict(payload["q_table"])
        return agent

    @classmethod
    def load(
        cls,
        model_id: str | Path,
        step: int,
        encoder: ObservationEncoder,
    ) -> "QTableAgent":
        return cls._load_from_path(
            path=model_path(model_id, step, ".pkl"),
            encoder=encoder,
        )
