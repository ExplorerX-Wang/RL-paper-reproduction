import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network import GaussianPolicy, QNetwork, SafetyNetwork,LambdaNetwork
from buffer import ReplayBuffer
from copy import deepcopy
from utils import calculate_f


class SSAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_bound,
        buffer_size=1e6,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        beta_q=8e-5,
        beta_pi=3e-5,
        beta_c=5e-5,
        beta_alpha=5e-5,
        beta_lambda=5e-5,
        policy_update_interval=3,
        lambda_update_interval=10,
        cost_limit=25.0,
        device=None,
        max_timesteps=1000000
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 初始化参数
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.policy_update_interval = policy_update_interval
        self.lambda_update_interval = lambda_update_interval
        self.cost_limit = cost_limit
        self.max_timesteps = max_timesteps
        
        # 学习率初始值和最终值
        self.beta_pi_start, self.beta_pi_end = 3e-5, 1e-6
        self.beta_q_start, self.beta_q_end = 8e-5, 1e-6
        self.beta_c_start, self.beta_c_end = 8e-5, 1e-6
        self.beta_alpha_start, self.beta_alpha_end = 5e-5, 1e-6
        self.beta_lambda_start, self.beta_lambda_end = 5e-5, 5e-6

        # 初始化网络
        self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.policy_target = deepcopy(self.policy).to(self.device)  # 添加目标策略网络
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = deepcopy(self.q_network).to(self.device)
        self.safety_network = SafetyNetwork(state_dim, action_dim).to(self.device)
        self.lambda_network = LambdaNetwork(state_dim).to(self.device)

        # 目标网络参数不需要梯度
        for param in self.q_target.parameters():
            param.requires_grad = False
        for param in self.policy_target.parameters():  # 目标策略网络参数不需要梯度
            param.requires_grad = False
            
        # 初始化优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=beta_pi)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=beta_q)
        self.safety_optimizer = optim.Adam(self.safety_network.parameters(), lr=beta_c)
        
        # 自动调整温度参数  
        self.target_entropy = -np.prod(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=beta_alpha)
        
        # 拉格朗日乘子优化器
        self.lambda_optimizer = optim.Adam(self.lambda_network.parameters(), lr=beta_lambda)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        
        # 训练步数计数器
        self.total_it = 0
        
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if deterministic:
        # 使用确定性策略
            action = self.policy.deterministic_action(state)
        else:
        # 使用重参数化采样
            action, _ = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0] * self.action_bound
    
    def _update_learning_rates(self):
        """更新所有优化器的学习率（线性退火）"""
        progress = min(self.total_it / self.max_timesteps, 1.0)
        
        # 计算当前学习率
        lr_pi = self.beta_pi_start + progress * (self.beta_pi_end - self.beta_pi_start)
        lr_q = self.beta_q_start + progress * (self.beta_q_end - self.beta_q_start)
        lr_c = self.beta_c_start + progress * (self.beta_c_end - self.beta_c_start)
        lr_alpha = self.beta_alpha_start + progress * (self.beta_alpha_end - self.beta_alpha_start)
        lr_lambda = self.beta_lambda_start + progress * (self.beta_lambda_end - self.beta_lambda_start)
        
        # 更新优化器学习率
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr_pi
        for param_group in self.q_optimizer.param_groups:
            param_group['lr'] = lr_q
        for param_group in self.safety_optimizer.param_groups:
            param_group['lr'] = lr_c
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = lr_alpha
        for param_group in self.lambda_optimizer.param_groups:
            param_group['lr'] = lr_lambda
            
        return {
            'lr_pi': lr_pi,
            'lr_q': lr_q,
            'lr_c': lr_c,
            'lr_alpha': lr_alpha,
            'lr_lambda': lr_lambda
        }

    def train(self):
        self.total_it += 1
        
        # 更新学习率
        lr_info = self._update_learning_rates()

        # 从缓冲区采样
        states, actions, rewards, costs, next_states, dones = self.replay_buffer.sample(self.batch_size)
        #print(f"reward:{rewards.mean()},cost:{costs.mean()}")

        current_q1, current_q2 = self.q_network(states, actions)
        # 更新Q网络
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_target.sample(next_states)  # 使用目标策略网络
            next_q1, next_q2 = self.q_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q

        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        #print("update q...")
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 更新安全网络
        current_c = self.safety_network(states, actions)
        #with torch.no_grad():
            #next_actions_target, _ = self.policy_target.sample(next_states)
            #next_c = self.safety_network(next_states, next_actions_target)
            #target_c = costs + (1 - dones) * self.gamma * next_c
            #target_c = f(next_states)- max{f(states), 0 }
            #target_c = costs

        c_loss = F.mse_loss(current_c, costs)
        #print("update c...")
        self.safety_optimizer.zero_grad()
        c_loss.backward()
        self.safety_optimizer.step()

        # 更新策略网络
        # new_actions, log_probs = self.policy.sample(states)
        # q1, q2 = self.q_network(states, new_actions)  # 使用新动作评估Q值
        # q = torch.min(q1, q2)
        #
        with torch.no_grad():
             lambda_value = self.lambda_network(states)  # 获取lambda值
        #
        # current_c = self.safety_network(states, new_actions)  # 使用新动作评估安全代价
        # # 初始化损失值
        policy_loss = torch.tensor(0.0, device=self.device)
        alpha_loss = torch.tensor(0.0, device=self.device)
        new_actions, log_probs = self.policy.sample(states)
        if self.total_it % self.policy_update_interval == 0:
            #print("update policy and alpha...")
            #print(f"alpha = {self.alpha}")
            #print(f"log_prob = {log_probs.mean()}")
            #print(f"q={q.mean()}")
            #print(f"lambda = {lambda_value.mean()}")
            #print(f"current_cost = {current_c.mean()}" )
            #new_actions, log_probs = self.policy.sample(states)
            q1, q2 = self.q_network(states, new_actions)  # 使用新动作评估Q值
            q = torch.min(q1, q2)
            current_c = self.safety_network(states, new_actions)  # 使用新动作评估安全代价


            policy_loss = (self.alpha * log_probs - q + lambda_value * current_c).mean()
            #print(f"policy_loss = {policy_loss}")
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()             
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # 更新lambda (拉格朗日乘子)
        if self.total_it % self.lambda_update_interval == 0:
            #print("update lambda ...")
            # 拉格朗日乘子更新基于约束违反情况
            with torch.no_grad():
                current_c_new = self.safety_network(states, new_actions.detach())
                #constraint_violation = current_c_new.mean() - self.cost_limit
                constraint_violation = current_c_new.mean()
            lambda_value_mean = self.lambda_network(states).mean()
            lambda_loss = -(lambda_value_mean * constraint_violation)
            
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()
        
        # 更新Q目标网络
        if self.total_it % (5 * self.policy_update_interval) == 0:
            for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # 更新策略目标网络
            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        avg_lambda = lambda_value.mean().item()
        return {
            'q_loss': q_loss.item(),
            'c_loss': c_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'lambda': avg_lambda
        }
    
    def save(self, filename):
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_target': self.policy_target.state_dict(),  # 保存目标策略网络
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'safety_network': self.safety_network.state_dict(),
            'lambda_network': self.lambda_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'safety_optimizer': self.safety_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'lambda_optimizer': self.lambda_optimizer.state_dict(),
            'total_it': self.total_it
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename)
        
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_target.load_state_dict(checkpoint['policy_target'])  # 加载目标策略网络
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.safety_network.load_state_dict(checkpoint['safety_network'])
        self.lambda_network.load_state_dict(checkpoint['lambda_network'])  # 加载lambda网络
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.safety_optimizer.load_state_dict(checkpoint['safety_optimizer'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.lambda_optimizer.load_state_dict(checkpoint['lambda_optimizer'])
        
        self.total_it = checkpoint['total_it']
        self.alpha = self.log_alpha.exp().item()