import safety_gymnasium
import numpy as np
import gymnasium

class CustomSafetyPointGoalEnv(gymnasium.Wrapper):
    def __init__(self):
        # 创建原始环境，不使用渲染模式
        env = safety_gymnasium.make('SafetyPointGoal1-v0',render_mode="human")
        super().__init__(env)
        
        # 设置参数
        self.sigma = 0.04
        self.d_min = 0.3
        self.k = 2
        self.n = 2
        
        # 保存上一时刻的f值
        self.last_f_value = None
        
    def _calculate_f(self, state, hazards_pos=None):
        """计算f(s)函数值: f(s) = sigma + (d_min)^n - d^n - k*d'
        
        参数:
        - state: 环境状态（observation）
        - hazards_pos: 障碍物位置信息（如果状态中不包含，则需要额外提供）
        """
        # 从状态中提取智能体位置和速度信息
        # 根据SafetyPointGoal1-v0的观察空间结构：
        # 索引3:6是速度计（速度）[vx, vy, vz]
        # 索引9:12是磁力计（位置）[x, y, z]
        agent_vel = state[3:6]    # 速度计读数对应速度信息 [vx, vy, vz]
        agent_pos = state[9:12]   # 磁力计读数对应位置信息 [x, y, z]
        
        # 如果没有提供hazards_pos，则从环境中获取
        if hazards_pos is None:
            hazards_pos = self.env.task.hazards.pos  # 障碍物位置
            
        # 确保hazards_pos是numpy数组
        hazards_pos = np.array(hazards_pos)
        
        if len(hazards_pos) == 0:
            # 没有障碍物，返回基础值
            return self.sigma + np.power(self.d_min, self.n)
        
        # 计算到所有障碍物的距离（只考虑2D平面距离）
        distances = np.linalg.norm(hazards_pos[:, :2] - agent_pos[:2], axis=1)
        
        # 找到最近的障碍物
        min_dist_idx = np.argmin(distances)
        d = distances[min_dist_idx]  # 到最近障碍物的距离
        
        # 计算相对速度（智能体速度朝向最近障碍物的分量）
        if d > 1e-8:  # 避免除零
            nearest_hazard_pos = hazards_pos[min_dist_idx]
            direction_to_hazard = (nearest_hazard_pos[:2] - agent_pos[:2]) / d
            d_prime = np.dot(agent_vel[:2], direction_to_hazard)  # 相对速度（只考虑2D平面）
        else:
            d_prime = 0.0
            d = 1e-8  # 避免数值问题
        
        # 计算f(s)
        f_value = (self.sigma + 
                   np.power(self.d_min, self.n) - 
                   np.power(d, self.n) - 
                   self.k * d_prime)
        
        return f_value
    
    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        # 重置上一时刻的f值
        self.last_f_value = None
        return observation, info
    
    def step(self, action):
        # 执行动作
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        
        # 计算当前时刻的f值
        current_f_value = self._calculate_f(observation)
        
        # 计算cost: c(s_t,a_t) = f(s_{t+1}) - max{f(s_t), 0}
        if self.last_f_value is not None:
            cost = current_f_value - max(self.last_f_value, 0)
        else:
            cost = current_f_value - max(0, 0)  # 第一步时last_f_value为None
            
        # 更新上一时刻的f值
        self.last_f_value = current_f_value
        
        return observation, reward, cost, terminated, truncated, info

if __name__ == "__main__":
    # 测试环境
    env = CustomSafetyPointGoalEnv()
    obs, info = env.reset()
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.3f}, Cost={cost:.3f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()