import gymnasium


def evaluate(agent, env: gymnasium.Env) -> float:
    observation, _ = env.reset()
    total_return = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.act(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
    return total_return
