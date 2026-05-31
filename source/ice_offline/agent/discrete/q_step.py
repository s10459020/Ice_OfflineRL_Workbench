from typing import Any

from .q_table import ObservationEncoder, QTableAgent


class StepQAgent:
    def __init__(
        self,
        n_actions: int,
        encoder: ObservationEncoder,
        alpha: float = 1.0,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self._step_counter = 0
        self._agent = QTableAgent(
            n_actions=n_actions,
            encoder=encoder,
            alpha=alpha,
            gamma=gamma,
            seed=seed,
        )

    def policy(self, observation: Any, epsilon: float = 0.0) -> int:
        return int(self._agent.policy(observation, epsilon=epsilon))

    def update(self, observation: Any, action: int) -> None:
        self._step_counter += 1
        self._agent.Q[observation][action] = float(self._step_counter)

    def Q(self, observation: Any, action: int) -> float:
        return float(self._agent.Q(observation, action))

    def q_states(self) -> int:
        return len(self._agent.Q)

    def step_count(self) -> int:
        return self._step_counter

