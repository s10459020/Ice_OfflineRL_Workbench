import gymnasium as gym


EVAL_EPISODES = 10


def evaluate(agent, env: gym.Env, dataset) -> float:
    observation, _ = env.reset()
    total_return = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.act(dataset.normalize_observation(observation))
        observation, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
    return total_return


def evaluate_mean(agent, env: gym.Env, dataset, episodes: int = EVAL_EPISODES) -> float:
    returns = [evaluate(agent, env, dataset) for _ in range(episodes)]
    return sum(returns) / len(returns)
