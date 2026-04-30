import gymnasium as gym


class StepPenaltyWrapper(gym.RewardWrapper):
    """Apply a fixed penalty on every step reward."""

    def __init__(self, env: gym.Env, step_penalty: float = 0.01) -> None:
        super().__init__(env)
        self.step_penalty = float(step_penalty)

    def reward(self, reward: float) -> float:
        return float(reward) - self.step_penalty
