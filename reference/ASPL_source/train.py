import argparse
import os
import sys
import numpy as np
import torch
import gym

from agent import ASPLPolicy, ReplayBuffer, evaluate_policy
from environments import make_env

def train(args):
    """Training function that can be called from main.py"""
    

    file_name = f"spl_{args.env}_{args.seed}"

    # ========== Step 1: Setup Environment and Directories ==========
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    env, max_action = make_env(args.env)

    # Set seeds
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ========== Step 2: Setup GPU Device ==========
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # ========== Step 3: Initialize Policy and Replay Buffer ==========
    policy = ASPLPolicy(
        state_dim,
        action_dim,
        max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
            alpha=args.alpha,
            num_sampled_actions=args.num_sampled_actions,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        max_timesteps=args.max_timesteps,
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load_d4rl_dataset(env)
    
    # ========== Step 4: Initial Evaluation and Setup ==========
    evaluations = []
    eval_score = evaluate_policy(policy, args.env, args.seed, replay_buffer, eval_episodes=args.eval_episodes)
    evaluations.append(eval_score)
    
    # ========== Step 5: Main Training Loop ==========
    print(f"Training for {args.max_timesteps} steps")
    for t in range(int(args.max_timesteps)):
        losses = policy.update(replay_buffer, args.batch_size)
        
        if (t + 1) % args.eval_freq == 0:
            eval_score = evaluate_policy(policy, args.env, args.seed, replay_buffer, eval_episodes=args.eval_episodes)
            evaluations.append(eval_score)
            
            
    
    # ========== Step 6: Final Evaluation and Cleanup ==========
    final_score = float(np.mean(evaluations[1:][-10:]))

    if args.save_model:
        policy.save(f"./saved_models/{file_name}")

    
    
    return policy, final_score 