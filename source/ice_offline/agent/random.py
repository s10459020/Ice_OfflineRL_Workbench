from typing import Any


class RandomAgent:
    def __init__(self, action_space, device: str = "cpu") -> None:
        self.action_space = action_space
        self.device = device

    def act_best(self, observation: Any):
        return self.action_space.sample()

