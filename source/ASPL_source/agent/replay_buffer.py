import numpy as np
import torch
import d4rl

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        
        self.state = np.empty((0, state_dim), dtype=np.float32)
        self.action = np.empty((0, action_dim), dtype=np.float32)
        self.next_state = np.empty((0, state_dim), dtype=np.float32)
        self.reward = np.empty((0, 1), dtype=np.float32)
        self.not_done = np.empty((0, 1), dtype=np.float32)
        
        self.size = 0

    def load_d4rl_dataset(self, env):
        """Load dataset from d4rl gym environment."""
        dataset = d4rl.qlearning_dataset(env)
        
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Buffer is empty! Call load_d4rl_dataset before sampling.")
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
 