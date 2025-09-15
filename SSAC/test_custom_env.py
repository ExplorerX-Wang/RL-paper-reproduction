import numpy as np
import safety_gymnasium
from custom_safety_env import CustomSafetyPointGoalEnv
from agent import SSAC
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env", default="CustomSafetyPointGoal1-v0", type=str, help="Safety Gymnasium环境名称")
    parser.add_argument("--seed", default=24, type=int, help="随机种子")

    # 训练参数
    parser.add_argument("--max_timesteps", default=10_000_000, type=int, help="最大训练步数")
    parser.add_argument("--start_timesteps", default=1000, type=int, help="开始训练前的随机采样步数")
    parser.add_argument("--batch_size", default=256, type=int, help="批量大小")
    # parser.add_argument("--eval_freq", default=10000, type=int, help="评估频率")
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
    parser.add_argument("--save_model", default='true', action="store_true", help="是否保存模型")
    parser.add_argument("--load_model", default="", type=str, help="加载模型路径")

    return parser.parse_args()



def test_custom_environment():
    """测试自定义环境"""
    print("Testing CustomSafetyPointGoalEnv...")

    args = parse_args()

    # 创建自定义环境
    env = CustomSafetyPointGoalEnv()

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



    # 重置环境
    state, _ = env.reset(seed=args.seed)
    print(f"Observation shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 运行几个步骤
    for i in range(10):
        # 随机动作
        action = agent.select_action(state,deterministic=False)
        print(action)
        obs, reward, cost, terminated, truncated, info = env.step(action)

        print(obs)
        print(f"Step {i+1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Cost: {cost:.4f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        
        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()
    
    # 关闭环境
    env.close()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_custom_environment()