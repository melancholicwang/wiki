"""
对比所有线性注意力实现的性能

运行此脚本可以直观对比：
1. 计算速度
2. 内存使用
3. 输出质量
"""

import torch
import time
import numpy as np
from standard_attention import StandardAttention
from mamba2 import Mamba2
from gated_linear_attention import GatedLinearAttention
from deltanet import DeltaNet
from gated_delta_network import GatedDeltaNetwork
from key_decomposed_attention import KeyDecomposedAttention


def measure_time_and_memory(model, x, num_runs=10, warmup=3):
    """测量模型的运行时间和内存使用"""
    device = x.device

    # 预热
    for _ in range(warmup):
        _ = model(x)

    # 清空缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 记录初始内存
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    # 测量时间
    start_time = time.time()

    for _ in range(num_runs):
        output = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs

    # 测量内存
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated()
        memory_used = (peak_mem - start_mem) / 1024 / 1024  # MB
    else:
        memory_used = 0

    return avg_time, memory_used, output


def compare_models():
    """对比所有模型"""
    print("=" * 80)
    print("线性注意力机制性能对比")
    print("=" * 80)

    # 配置
    batch_size = 4
    d_model = 256
    num_heads = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n设备: {device}")
    print(f"配置: batch_size={batch_size}, d_model={d_model}, num_heads={num_heads}")

    # 不同序列长度
    seq_lengths = [128, 256, 512, 1024]

    # 创建模型
    models = {
        'Standard Attention': StandardAttention(d_model, num_heads).to(device),
        'Mamba2': Mamba2(d_model, d_state=16).to(device),
        'GLA': GatedLinearAttention(d_model, num_heads).to(device),
        'DeltaNet': DeltaNet(d_model, num_heads).to(device),
        'GDN': GatedDeltaNetwork(d_model, num_heads).to(device),
        'KDA (Mixed)': KeyDecomposedAttention(
            d_model, num_heads, num_subspaces=4, subspace_type="mixed"
        ).to(device),
    }

    # 设置为评估模式
    for model in models.values():
        model.eval()

    # 对每个序列长度进行测试
    results = {name: {'time': [], 'memory': []} for name in models.keys()}

    for seq_len in seq_lengths:
        print(f"\n{'='*80}")
        print(f"序列长度: {seq_len}")
        print(f"{'='*80}")
        print(f"{'模型':<20} {'时间 (ms)':<15} {'内存 (MB)':<15} {'相对速度':<15}")
        print("-" * 80)

        # 生成输入
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        baseline_time = None
        current_results = {}

        for name, model in models.items():
            try:
                with torch.no_grad():
                    avg_time, memory_used, output = measure_time_and_memory(
                        model, x, num_runs=10
                    )

                # 记录结果
                results[name]['time'].append(avg_time * 1000)  # 转换为 ms
                results[name]['memory'].append(memory_used)
                current_results[name] = (avg_time, memory_used)

                # 设置基线（Standard Attention）
                if baseline_time is None:
                    baseline_time = avg_time

                relative_speed = baseline_time / avg_time

                print(f"{name:<20} {avg_time*1000:>10.2f} ms   "
                      f"{memory_used:>10.2f} MB   {relative_speed:>10.2f}x")

            except Exception as e:
                print(f"{name:<20} ERROR: {str(e)[:40]}")
                results[name]['time'].append(float('inf'))
                results[name]['memory'].append(float('inf'))

    # 绘制总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")

    print("\n平均加速比（相对于 Standard Attention）：")
    print("-" * 80)

    for name in models.keys():
        if name == 'Standard Attention':
            continue

        valid_times = [t for t in results[name]['time'] if t != float('inf')]
        valid_baseline = [results['Standard Attention']['time'][i]
                         for i, t in enumerate(results[name]['time'])
                         if t != float('inf')]

        if valid_times and valid_baseline:
            avg_speedup = np.mean([b/t for b, t in zip(valid_baseline, valid_times)])
            print(f"{name:<20} {avg_speedup:>10.2f}x")

    print("\n平均内存使用（MB）：")
    print("-" * 80)

    for name in models.keys():
        valid_memory = [m for m in results[name]['memory'] if m != float('inf')]
        if valid_memory:
            avg_memory = np.mean(valid_memory)
            print(f"{name:<20} {avg_memory:>10.2f} MB")

    # 理论复杂度提醒
    print(f"\n{'='*80}")
    print("理论复杂度对比（T=序列长度，D=模型维度）")
    print(f"{'='*80}")

    complexities = {
        'Standard Attention': 'O(T² × D)',
        'Mamba2': 'O(T × D × N), N=状态维度',
        'GLA': 'O(T × D²)',
        'DeltaNet': 'O(T × D²)',
        'GDN': 'O(T × D²)',
        'KDA (Mixed)': 'O(T² × D/n + T × D²/n), n=子空间数',
    }

    for name, complexity in complexities.items():
        print(f"{name:<20} {complexity}")

    print(f"\n{'='*80}")
    print("推荐使用场景")
    print(f"{'='*80}")

    recommendations = {
        'Standard Attention': '短序列 (<512), 需要最强表达能力',
        'Mamba2': '超长序列 (>10K), 需要极致效率',
        'GLA': '中长序列 (512-5K), 平衡实现和性能',
        'DeltaNet': '位置敏感任务（如代码生成）',
        'GDN': '复杂任务，需要强表达能力',
        'KDA (Mixed)': '需要灵活性，多模态任务',
    }

    for name, rec in recommendations.items():
        print(f"\n{name}:")
        print(f"  → {rec}")


def visualize_attention_patterns():
    """可视化不同注意力机制的模式"""
    print(f"\n{'='*80}")
    print("注意力模式分析")
    print(f"{'='*80}")

    batch_size, seq_len, d_model = 1, 64, 64
    num_heads = 4

    device = torch.device('cpu')  # 使用 CPU 以便提取注意力权重

    # 创建简单的输入（带模式）
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # 标准注意力（可以提取注意力权重）
    sa = StandardAttention(d_model, num_heads).to(device)
    sa.eval()

    with torch.no_grad():
        output, attn_weights = sa(x)

    print(f"\n标准注意力权重形状: {attn_weights.shape}")
    print(f"注意力权重统计:")
    print(f"  最小值: {attn_weights.min().item():.6f}")
    print(f"  最大值: {attn_weights.max().item():.6f}")
    print(f"  均值: {attn_weights.mean().item():.6f}")
    print(f"  每行和: {attn_weights.sum(dim=-1).mean().item():.6f} (应该≈1.0)")

    # 分析稀疏性
    threshold = 0.01
    sparse_ratio = (attn_weights < threshold).float().mean().item()
    print(f"  稀疏度 (<{threshold}): {sparse_ratio*100:.1f}%")

    print("\n注意：其他线性注意力方法不显式计算完整注意力矩阵，")
    print("因此无法直接可视化，但这正是它们高效的原因！")


def test_output_similarity():
    """测试不同方法的输出相似度"""
    print(f"\n{'='*80}")
    print("输出相似度分析")
    print(f"{'='*80}")

    batch_size, seq_len, d_model = 2, 128, 64
    num_heads = 8
    device = torch.device('cpu')

    # 固定随机种子
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # 创建模型
    models = {
        'Standard Attention': StandardAttention(d_model, num_heads).to(device),
        'GLA': GatedLinearAttention(d_model, num_heads).to(device),
        'DeltaNet': DeltaNet(d_model, num_heads).to(device),
        'GDN': GatedDeltaNetwork(d_model, num_heads).to(device),
    }

    # 获取输出
    outputs = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if name == 'Standard Attention':
                output, _ = model(x)
            else:
                output = model(x)
            outputs[name] = output

    # 计算相似度（余弦相似度）
    print("\n输出余弦相似度矩阵（与 Standard Attention 对比）：")
    print("-" * 80)

    baseline = outputs['Standard Attention']
    baseline_flat = baseline.reshape(-1)

    for name, output in outputs.items():
        if name == 'Standard Attention':
            continue

        output_flat = output.reshape(-1)
        similarity = torch.nn.functional.cosine_similarity(
            baseline_flat.unsqueeze(0),
            output_flat.unsqueeze(0)
        ).item()

        print(f"{name:<20} 相似度: {similarity:.4f}")

    print("\n注意：相似度较低不一定意味着性能差，")
    print("线性注意力方法使用不同的归纳偏置，可能在某些任务上更好！")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行对比
    try:
        compare_models()
    except Exception as e:
        print(f"\n性能对比出错: {e}")

    # 可视化注意力模式
    try:
        visualize_attention_patterns()
    except Exception as e:
        print(f"\n注意力模式分析出错: {e}")

    # 输出相似度测试
    try:
        test_output_similarity()
    except Exception as e:
        print(f"\n相似度分析出错: {e}")

    print(f"\n{'='*80}")
    print("对比完成！")
    print(f"{'='*80}")
    print("\n更多信息请参考 README.md")
