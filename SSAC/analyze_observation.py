import safety_gymnasium
import numpy as np

# 创建环境实例来检查观察空间结构
env = safety_gymnasium.make('SafetyPointGoal1-v0')

# 重置环境获取初始观察值
obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Observation:", obs)

# 让我们分析观察空间的结构
print("\n观察空间组件分析:")
print("1. Accelerometer (3维):", obs[0:3])  # 加速度计读数
print("2. Velocimeter (3维):", obs[3:6])    # 速度计读数
print("3. Gyro (3维):", obs[6:9])           # 陀螺仪读数
print("4. Magnetometer (3维):", obs[9:12])  # 磁力计读数
print("5. Goal Lidar (16维):", obs[12:28])  # 目标lidar读数
print("6. Hazards Lidar (16维):", obs[28:44])  # 障碍物lidar读数
print("7. Vases Lidar (16维):", obs[44:60])  # 花瓶lidar读数

# 修正索引
print("\n修正后的索引:")
print("位置信息 (磁力计):", obs[9:12])  # 索引9到11
print("速度信息 (速度计):", obs[3:6])   # 索引3到5

env.close()