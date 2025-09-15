# **Reference Paper：**

[1].Ma H, Liu C, Li S E, et al. Learn zero-constraint-violation safe policy in model-free constrained reinforcement learning[J]. IEEE Transactions on Neural Networks and Learning Systems, 2024.

[2].Zhao W, He T, Liu C. Model-free safe control for zero-violation reinforcement learning[C]//5th Annual Conference on Robot Learning. 2021.


# **Requirements:**

mujoco==2.3.0

numpy==1.23.5

torch==2.4.0+cu124

gymnasium==0.28.1

safety-gymnasium==1.0.0

matplotlib==3.10.6

# **Notes:**

1. cost函数使用论文中的公式（see custom_safety_env.py）,但是输出的cost总是负数，需修改
2. 使用了SafetyPointGoal1-v0环境默认的奖励函数,防止智能体为了安全而原地不动，对奖励放大了100倍
3. 拉格朗日乘子网络输出嵌位在(0, 1000),因为之前遇到过这样的问题：随着训练进行，拉格朗日乘子持续发散（怀疑和代码中cost总为负数有关，见论文[1]）
4. 代码实现的效果：point可以快速到到绿色区域，但没有避开蓝色区域(Hazards)的行为

# **TODO：**
1. SafetyPointGoal1-v0环境，状态变量(60维)的含义是什么？ 如何获取障碍物和目标的位置？ 
2. cost的计算：怎么在SafetyPointGoal1-v0环境下，由状态变量计算safety index？