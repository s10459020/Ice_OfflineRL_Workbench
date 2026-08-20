from dataclasses import dataclass

import gymnasium as gym

from joint_learning.lib.eval import evaluate_mean


@dataclass
class TestResult:
    mean_return: float


def test(agent, dataset, eval_count: int) -> TestResult:
    env = gym.make(dataset.env_id)
    mean_return = evaluate_mean(agent, env, dataset, eval_count)
    env.close()
    return TestResult(mean_return)
