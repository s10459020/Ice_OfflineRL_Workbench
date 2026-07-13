import gym
import warnings

def make_env(env_name):
    """
    Create a D4RL environment.
    
    Args:
        env_name (str): Name of the D4RL environment to create.
        
    Returns:
        env: Gym environment instance
    """
    # Suppress Box bound precision warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Box bound precision lowered by casting to float32")
        env = gym.make(env_name)
    
    # Get environment action space bounds
    max_action = float(env.action_space.high[0])
    
    return env, max_action 