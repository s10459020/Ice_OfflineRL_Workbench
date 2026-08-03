import gymnasium as gym
import numpy as np

from joint_learning.agent import make_agent
from joint_learning.agent import model_path
from joint_learning.dataset import load_dataset_info


# Keep these values aligned with train.py when evaluating a saved final model.
AGENT = "td3bc"
DATASET = "hopper_medium"
DEVICE = "cuda"
EPISODES = 10


def evaluate(agent, env: gym.Env, episodes: int) -> list[float]:
    returns: list[float] = []
    for _ in range(episodes):
        observation, _ = env.reset()
        total_return = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
        returns.append(total_return)
    return returns


def main() -> None:
    env_id, obs_size, act_size = load_dataset_info(DATASET)
    agent = make_agent(AGENT, obs_size, act_size, DEVICE)
    path = model_path(AGENT, DATASET)
    agent.load(path)

    env = gym.make(env_id)
    returns = evaluate(agent, env, EPISODES)
    env.close()

    values = np.asarray(returns, dtype=np.float32)
    print(f"eval agent={AGENT} dataset={DATASET} model={path}")
    print(
        f"episodes={EPISODES} "
        f"mean_return={values.mean():.6g} "
        f"std_return={values.std():.6g} "
        f"min_return={values.min():.6g} "
        f"max_return={values.max():.6g}"
    )


if __name__ == "__main__":
    main()
