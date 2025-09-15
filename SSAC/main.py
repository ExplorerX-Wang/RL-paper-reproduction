import numpy as np
import torch
#import gymnasium as gym
import safety_gymnasium
import argparse
import os
from agent import SSAC
from utils import evaluate_policy, plot_learning_curve, set_seed
# 导入自定义环境
from custom_safety_env import CustomSafetyPointGoalEnv

TRAIN_FLAG = False

def parse_args():
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env", default="CustomSafetyPointGoal1-v0", type=str, help="Safety Gymnasium环境名称")
    parser.add_argument("--seed", default=1024, type=int, help="随机种子")
    
    # 训练参数
    parser.add_argument("--max_timesteps", default=10_000_000, type=int, help="最大训练步数")
    parser.add_argument("--start_timesteps", default=1000, type=int, help="开始训练前的随机采样步数")
    parser.add_argument("--batch_size", default=256, type=int, help="批量大小")
    #parser.add_argument("--eval_freq", default=10000, type=int, help="评估频率")
    parser.add_argument("--save_freq", default=50000, type=int, help="保存模型频率")
    
    # 算法参数
    parser.add_argument("--discount", default=0.99, type=float, help="折扣因子")
    parser.add_argument("--tau", default=0.005, type=float, help="目标网络软更新系数")
    parser.add_argument("--alpha", default=0.2, type=float, help="初始温度参数")
    parser.add_argument("--cost_limit", default=25.0, type=float, help="安全约束上限")
    
    # 学习率
    parser.add_argument("--lr_q", default=8e-5, type=float, help="Q网络学习率")
    parser.add_argument("--lr_pi", default=3e-5, type=float, help="策略网络学习率")
    parser.add_argument("--lr_c", default=8e-5, type=float, help="安全网络学习率")
    parser.add_argument("--lr_alpha", default=5e-5, type=float, help="温度参数学习率")
    parser.add_argument("--lr_lambda", default=5e-5, type=float, help="拉格朗日乘子学习率")
    
    # 其他参数
    parser.add_argument("--policy_update_interval", default=3, type=int, help="策略网络更新间隔")
    parser.add_argument("--lambda_update_interval", default=10, type=int, help="拉格朗日乘子更新间隔")
    parser.add_argument("--save_model", default = 'true', action="store_true", help="是否保存模型")
    parser.add_argument("--load_model", default="", type=str, help="加载模型路径")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建结果目录
    results_dir = f"./results/{args.env}/{args.seed}"
    models_dir = f"./models/{args.env}/{args.seed}"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if args.save_model and not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建环境
    if args.env == "CustomSafetyPointGoal1-v0":
        env = CustomSafetyPointGoalEnv()
        eval_env = CustomSafetyPointGoalEnv()
    else:
        env = safety_gymnasium.make(args.env, render_mode='human')
        eval_env = safety_gymnasium.make(args.env, render_mode='human')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    
    # 创建智能体
    agent = SSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        buffer_size=1e6,
        batch_size=args.batch_size,
        gamma=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        beta_q=args.lr_q,
        beta_pi=args.lr_pi,
        beta_c=args.lr_c,
        beta_alpha=args.lr_alpha,
        beta_lambda=args.lr_lambda,
        policy_update_interval=args.policy_update_interval,
        lambda_update_interval=args.lambda_update_interval,
        cost_limit=args.cost_limit,
        max_timesteps=10000000
    )

    # 加载模型（如果指定）
    if args.load_model != "":
        agent.load(args.load_model)
    
    # 初始化记录数据
    evaluations_rewards = []
    evaluations_costs = []
    episode_training_records = {
        'rewards': [],
        'costs': []
    }
    training_records = {  
        'q_loss': [],
        'c_loss': [],
        'policy_loss': [],
        'alpha_loss': [],
        'alpha': [],
        'lambda': []
    }
    # 初始化episode记录数据
    episode_rewards = []
    episode_costs = []

    state, _ = env.reset(seed=args.seed)
    #done = False
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0
    
    # 主训练循环
    for t in range(args.max_timesteps):
        episode_timesteps += 1

        # 使用策略选择动作
        action = agent.select_action(state,deterministic=False)
        #print(state)
        # 执行动作
        next_state, reward, cost, terminated, truncated, info = env.step(action)

        reward = 100 * reward

        #print(reward)
        done = terminated or truncated
        
        # 存储转换
        agent.replay_buffer.add(state, action, reward, cost, next_state, float(done))
        
        state = next_state
        episode_reward += reward
        episode_cost += cost
        
        # 训练智能体
        if t >= args.start_timesteps:
            #print("开始训练....")
            train_record = agent.train()
            
            # 记录训练数据
            for key in training_records:
                training_records[key].append(train_record[key])
        
        # 如果episode结束
        if done:
            print(f"Episode {episode_num+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, Steps = {episode_timesteps}")

            # 计算平均reward和cost
            avg_episode_reward = episode_reward / episode_timesteps
            avg_episode_cost = episode_cost / episode_timesteps

            # 记录每个episode的总reward和cost
            episode_training_records['rewards'].append(avg_episode_reward)
            episode_training_records['costs'].append(avg_episode_cost)

              # 记录每个episode的平均reward和cost
            episode_rewards.append(avg_episode_reward)
            episode_costs.append(avg_episode_cost)

            # 每10个episode保存一次数据
            if (episode_num + 1) % 10 == 0:
                # 保存episode平均reward和cost
                np.save(f"{results_dir}/episode_avg_rewards.npy", np.array(episode_rewards))
                np.save(f"{results_dir}/episode_avg_costs.npy", np.array(episode_costs))
                
                # 保存训练记录
                np.save(f"{results_dir}/training_records.npy", training_records)
                np.save(f"{results_dir}/episode_training_records.npy", episode_training_records)
                print(f"Episode {episode_num+1}: 数据已保存")


            # 重置环境
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            episode_num += 1
        
        # 保存模型
        if args.save_model and (t + 1) % args.save_freq == 0:
            agent.save(f"{models_dir}/{t+1}.pt")
    
    # 训练结束，保存最终模型
    if args.save_model:
        agent.save(f"{models_dir}/final.pt")

if __name__ == "__main__":
    main()