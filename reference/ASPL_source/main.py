import argparse
from train import train
# pyright: reportUndefinedVariable=false

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # ========== Environment and Basic Setup ==========
    parser.add_argument("--env", default="hopper-medium-v2", help="D4RL environment name") 
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # ========== Training Schedule ==========
    parser.add_argument("--max_timesteps", type=int, default=1000000, help="Maximum training timesteps")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation during training")
    parser.add_argument("--final_eval_episodes", type=int, default=100, help="Number of episodes for final evaluation")
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    
    # ========== Network Architecture ==========
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers")
    
    
    # ========== Optimization Parameters ==========
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    
    # ========== TD3 Policy Parameters ==========
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Policy noise for target")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Noise clip for target")
    parser.add_argument("--policy_freq", type=int, default=2, help="Policy update frequency")
    
    # ========== Data Preprocessing ==========
    
    # ========== Logging Parameters ==========
    
    
    # ========== SPL-Specific Parameters ==========
    parser.add_argument("--alpha", type=float, default=0.05, help="SPL alpha parameter")
    parser.add_argument("--num_sampled_actions", type=int, default=6, help="Number of actions to sample for unsupervised learning")
    

 
    
    args = parser.parse_args()
    
    # Sampling method is fixed to LHS inside the algorithm; no CLI options required.
    
    train(args)
