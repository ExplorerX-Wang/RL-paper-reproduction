import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import softplus


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256, 256), activation=nn.ELU):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            
        self.activation = activation()
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256, 256), activation=nn.ELU):
        super(GaussianPolicy, self).__init__()
        
        self.net = MLP(state_dim, 2 * action_dim, hidden_dims, activation)
        self.action_dim = action_dim
        
    def forward(self, state):
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # 重参数化采样
        
        # Squash to [-1, 1]
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256, 256), activation=nn.ELU):
        super(QNetwork, self).__init__()
        
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dims, activation)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dims, activation)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2

class SafetyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256, 256), activation=nn.ELU):
        super(SafetyNetwork, self).__init__()
        
        self.c = MLP(state_dim + action_dim, 1, hidden_dims, activation)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.c(x)

class LambdaNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=(256, 256, 256), activation=nn.ELU):
        super(LambdaNetwork, self).__init__()
        
        self.net = MLP(state_dim, 1, hidden_dims, activation)
        
    def forward(self, state):
        # 使用softplus激活函数确保输出为非负数
        softplus_out = F.softplus(self.net(state))
        return torch.clamp(softplus_out,min=0,max=1000)