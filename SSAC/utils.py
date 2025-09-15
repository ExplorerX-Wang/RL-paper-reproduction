import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def evaluate_policy(env, policy, eval_episodes=10, render=False):
    avg_reward = 0.
    avg_cost = 0.
    
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        
        while not done:
            action = policy.select_action(state, evaluate=True)
            next_state, reward, cost, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_cost += cost
            state = next_state
            
            if render:
                env.render()
                
        avg_reward += episode_reward
        avg_cost += episode_cost
        
    avg_reward /= eval_episodes
    avg_cost /= eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, Cost: {avg_cost:.3f}")
    print("---------------------------------------")
    
    return avg_reward, avg_cost

def plot_learning_curve(x, y_list, title, legend_list, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    
    for y, legend in zip(y_list, legend_list):
        plt.plot(x, y, label=legend)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename)
    plt.close()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def calculate_f(state, env, sigma, k, n, d_min, hazards_pos=None):
    """计算f(s)函数值: f(s) = sigma + (d_min)^n - d^n - k*d'

    参数:
    - state: 环境状态（observation）
    - hazards_pos: 障碍物位置信息（如果状态中不包含，则需要额外提供）
    """
    # 从状态中提取智能体位置和速度信息
    # 根据SafetyPointGoal1-v0的观察空间结构：
    # 索引3:6是速度计（速度）[vx, vy, vz]
    # 索引9:12是磁力计（位置）[x, y, z]
    agent_vel = state[3:6]  # 速度计读数对应速度信息 [vx, vy, vz]
    agent_pos = state[9:12]  # 磁力计读数对应位置信息 [x, y, z]

    # 如果没有提供hazards_pos，则从环境中获取
    if hazards_pos is None:
        hazards_pos = env.task.hazards.pos  # 障碍物位置

    # 确保hazards_pos是numpy数组
    hazards_pos = np.array(hazards_pos)

    if len(hazards_pos) == 0:
        # 没有障碍物，返回基础值
        return sigma + np.power(d_min, n)

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
    f_value = (sigma +
               np.power(d_min, n) -
               np.power(d, n) -
               k * d_prime)

    return f_value

