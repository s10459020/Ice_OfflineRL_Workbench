import numpy as np
import gym
import sys
import os

# Add parent directory to Python path to find agent and environments modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments import make_env

def evaluate_policy(policy, env_name, seed, replay_buffer, eval_episodes=10):
    """
    Evaluates the policy over a number of episodes and returns the average score.
    """
    eval_env, _ = make_env(env_name)
    eval_env.reset(seed=seed)

    avg_reward = 0.
    for i in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    
    return d4rl_score 