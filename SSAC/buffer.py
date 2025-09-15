import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=500000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((self.max_size, state_dim))
        self.actions = np.zeros((self.max_size, action_dim))
        self.rewards = np.zeros((self.max_size, 1))
        self.costs = np.zeros((self.max_size, 1))
        self.next_states = np.zeros((self.max_size, state_dim))
        self.dones = np.zeros((self.max_size, 1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, reward, cost, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.costs[self.ptr] = cost
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.costs[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )