import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def smooth(data, window=10):
    """使用滑动窗口平滑数据"""
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

def plot_curves(data_path, save_dir, window_size=10):
    """绘制训练曲线"""
    # 加载训练记录
    try:
        records = np.load(data_path, allow_pickle=True).item()
    except:
        print(f"无法加载数据文件: {data_path}")
        return
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制奖励曲线
    if 'rewards' in records:
        rewards = np.array(records['rewards'])
        if len(rewards) > window_size:
            smoothed_rewards = smooth(rewards, window=window_size)
            
            plt.figure(figsize=(10, 6))
            plt.plot(rewards, alpha=0.3, color='blue', label='原始奖励')
            plt.plot(np.arange(window_size-1, len(rewards)), smoothed_rewards, 
                    color='blue', label=f'平滑奖励 (窗口={window_size})')
            plt.xlabel('训练步数')
            plt.ylabel('奖励')
            plt.title('训练奖励曲线')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{save_dir}/reward_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"奖励曲线已保存至 {save_dir}/reward_curve.png")
    
    # 绘制代价曲线
    if 'costs' in records:
        costs = np.array(records['costs'])
        if len(costs) > window_size:
            smoothed_costs = smooth(costs, window=window_size)
            
            plt.figure(figsize=(10, 6))
            plt.plot(costs, alpha=0.3, color='red', label='原始代价')
            plt.plot(np.arange(window_size-1, len(costs)), smoothed_costs, 
                    color='red', label=f'平滑代价 (窗口={window_size})')
            plt.axhline(y=25.0, color='black', linestyle='--', label='代价限制')
            plt.xlabel('训练步数')
            plt.ylabel('代价')
            plt.title('训练代价曲线')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{save_dir}/cost_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"代价曲线已保存至 {save_dir}/cost_curve.png")
    
    # 绘制损失曲线
    loss_keys = ['q_loss', 'c_loss', 'policy_loss', 'alpha_loss']
    for key in loss_keys:
        if key in records:
            values = np.array(records[key])
            if len(values) > window_size:
                smoothed_values = smooth(values, window=window_size)
                
                plt.figure(figsize=(10, 6))
                plt.plot(values, alpha=0.3, color='green', label=f'原始{key}')
                plt.plot(np.arange(window_size-1, len(values)), smoothed_values, 
                        color='green', label=f'平滑{key} (窗口={window_size})')
                plt.xlabel('训练步数')
                plt.ylabel(key)
                plt.title(f'训练{key}曲线')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(f"{save_dir}/{key}_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"{key}曲线已保存至 {save_dir}/{key}_curve.png")
    
    # 绘制alpha和lambda曲线
    param_keys = ['alpha', 'lambda']
    for key in param_keys:
        if key in records:
            values = np.array(records[key])
            if len(values) > window_size:
                smoothed_values = smooth(values, window=window_size)
                
                plt.figure(figsize=(10, 6))
                plt.plot(values, alpha=0.3, color='purple', label=f'原始{key}')
                plt.plot(np.arange(window_size-1, len(values)), smoothed_values, 
                        color='purple', label=f'平滑{key} (窗口={window_size})')
                plt.xlabel('训练步数')
                plt.ylabel(key)
                plt.title(f'训练{key}曲线')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(f"{save_dir}/{key}_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"{key}曲线已保存至 {save_dir}/{key}_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--data_path', type=str, default='results/training_records.npy', 
                        help='训练记录数据路径')
    parser.add_argument('--save_dir', type=str, default='results/plots', 
                        help='图表保存目录')
    parser.add_argument('--window_size', type=int, default=10, 
                        help='平滑窗口大小')
    
    args = parser.parse_args()
    plot_curves(args.data_path, args.save_dir, args.window_size)