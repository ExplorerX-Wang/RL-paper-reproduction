import numpy as np
import os
# 禁用渲染
os.environ['MUJOCO_GL'] = 'egl'

from custom_safety_env import CustomSafetyPointGoalEnv

def test_custom_environment():
    """测试自定义环境"""
    print("Testing CustomSafetyPointGoalEnv...")
    
    # 创建自定义环境
    env = CustomSafetyPointGoalEnv()
    
    # 重置环境
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 运行几个步骤
    for i in range(10):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        
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