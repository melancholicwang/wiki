#!/usr/bin/env python3
"""
估算MDLM (Masked Diffusion Language Model)所需的计算资源

基于论文 "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
参考：https://github.com/kuleshov-group/mdlm

核心思想：将RoBERTa的MLM（Masked Language Modeling）视为一种离散扩散过程
- 训练：加权的MLM损失，覆盖不同的masking rates
- 推理：迭代demasking过程，逐步恢复完整序列

计算内容：
- FLOPs (浮点运算次数)
- 参数量
- 内存需求
- 训练时间
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ModelConfig:
    """MDLM模型配置

    MDLM使用标准的encoder-only架构（类似BERT/RoBERTa），
    不需要复杂的时间嵌入网络
    """
    name: str
    vocab_size: int = 50265  # RoBERTa vocab size
    max_seq_len: int = 1024  # MDLM通常支持更长序列
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_ratio: float = 4.0

    # MDLM采样参数
    sampling_steps: int = 1000  # 采样/去噪步数（可调：1000-10000）

    def __post_init__(self):
        self.d_head = self.n_embd // self.n_head
        self.d_mlp = int(self.n_embd * self.mlp_ratio)


# 预定义的MDLM模型配置
# 基于MDLM论文的标准配置
MDLM_CONFIGS = {
    "mdlm-small": ModelConfig(
        name="mdlm-small",
        n_layer=12,
        n_head=12,
        n_embd=768,
        max_seq_len=1024,
    ),
    "mdlm-base": ModelConfig(
        name="mdlm-base",
        n_layer=12,
        n_head=12,
        n_embd=768,
        max_seq_len=1024,
    ),
    "mdlm-large": ModelConfig(
        name="mdlm-large",
        n_layer=24,
        n_head=16,
        n_embd=1024,
        max_seq_len=1024,
    ),
    "mdlm-xlarge": ModelConfig(
        name="mdlm-xlarge",
        n_layer=36,
        n_head=20,
        n_embd=1280,
        max_seq_len=1024,
    ),
}


def estimate_parameters(config: ModelConfig) -> Dict[str, int]:
    """
    估算MDLM模型参数量

    MDLM使用标准的encoder-only架构（类似BERT/RoBERTa）：
    - Token embedding
    - Position embedding
    - Transformer encoder layers (self-attention + MLP)
    - Output projection head

    注意：MDLM不需要复杂的时间嵌入网络，
    时间步信息可以通过简单的位置编码或额外的token来编码
    """
    params = {}

    # Token embedding: vocab_size × n_embd
    params['token_embd'] = config.vocab_size * config.n_embd

    # Position embedding: max_seq_len × n_embd
    params['pos_embd'] = config.max_seq_len * config.n_embd

    # 每个Transformer层的参数
    # Self-attention: Q, K, V, O 投影矩阵
    params['attn_qkv_proj_per_layer'] = 3 * config.n_embd * config.n_embd  # Q, K, V
    params['attn_out_proj_per_layer'] = config.n_embd * config.n_embd  # O

    # Layer norm (2 per layer): 2 × (scale + bias)
    params['ln_per_layer'] = 2 * 2 * config.n_embd

    # MLP: two linear layers (with GELU activation)
    params['mlp_per_layer'] = (config.n_embd * config.d_mlp +  # up projection + bias
                               config.d_mlp +                   # bias
                               config.d_mlp * config.n_embd +   # down projection
                               config.n_embd)                   # bias

    # 所有层的总参数
    params_per_layer = (params['attn_qkv_proj_per_layer'] +
                       params['attn_out_proj_per_layer'] +
                       params['ln_per_layer'] +
                       params['mlp_per_layer'])

    params['all_layers'] = params_per_layer * config.n_layer

    # 最终的LayerNorm
    params['final_ln'] = 2 * config.n_embd

    # Output projection head: n_embd → vocab_size
    # (可以与token embedding共享权重，这里按独立计算)
    params['output_head'] = config.n_embd * config.vocab_size + config.vocab_size

    # 总参数量
    params['total'] = sum(params.values())

    return params


def estimate_flops_per_token(config: ModelConfig,
                             is_training: bool = True) -> int:
    """
    估算每个token的FLOPs

    参考nanoGPT的计算方法：
    - 前向传播: ~6N FLOPs per token (N是参数量)
    - 反向传播: ~2倍前向传播

    对于MDLM：训练时的计算量与标准BERT/RoBERTa MLM训练基本相同
    """
    params = estimate_parameters(config)
    N = params['total']

    # Encoder-only Transformer的前向传播: ~6N per token
    # 包括：2N (matmul) + 4N (attention)
    forward_flops_per_token = 6 * N

    if is_training:
        # 训练时需要前向+反向传播
        # 反向传播约为前向的2倍
        flops_per_token = forward_flops_per_token * 3  # forward + backward
    else:
        flops_per_token = forward_flops_per_token

    return flops_per_token


def estimate_mdlm_sampling_flops(config: ModelConfig,
                                 batch_size: int,
                                 seq_len: int,
                                 num_samples: int,
                                 sampling_steps: int = None) -> Dict[str, float]:
    """
    估算MDLM采样/生成的总FLOPs

    关键洞察：MDLM的迭代demasking过程中，
    - 每步只需要预测masked positions的tokens
    - 随着采样进行，masked tokens数量递减
    - 但仍需要对整个序列做前向传播（因为是encoder架构）

    采样过程：
    1. 开始时所有tokens都是[MASK]
    2. 每步unmask一部分tokens
    3. T步后完成生成

    参数：
        sampling_steps: 采样步数，默认使用config中的值
    """
    if sampling_steps is None:
        sampling_steps = config.sampling_steps

    flops_per_token = estimate_flops_per_token(config, is_training=False)

    # 推理时：需要多步迭代demasking
    # 每步都要对整个序列做前向传播
    num_forward_passes = sampling_steps

    total_tokens = batch_size * seq_len * num_samples
    total_flops = flops_per_token * total_tokens * num_forward_passes

    return {
        'total_flops': total_flops,
        'total_tflops': total_flops / 1e12,
        'total_pflops': total_flops / 1e15,
        'flops_per_token': flops_per_token,
        'sampling_steps': sampling_steps,
        'total_tokens': total_tokens,
        'tokens_per_step': total_tokens,
    }


def estimate_memory(config: ModelConfig,
                   batch_size: int,
                   seq_len: int,
                   dtype_bytes: int = 2) -> Dict[str, float]:
    """
    估算内存需求（单位：GB）

    包括：
    - 模型参数
    - 梯度
    - 优化器状态（AdamW需要2倍参数量）
    - 激活值
    """
    params = estimate_parameters(config)
    N = params['total']

    # 模型参数（FP16/BF16: 2 bytes per param）
    model_memory = N * dtype_bytes / 1e9

    # 训练时的额外内存
    # 梯度: 与参数相同大小
    gradient_memory = model_memory

    # 优化器状态（AdamW: momentum + variance）
    optimizer_memory = 2 * N * 4 / 1e9  # 优化器状态通常用FP32

    # 激活值（与batch size成正比）
    # 粗略估计：每层约保存 batch_size × seq_len × n_embd 的激活
    activation_per_layer = batch_size * seq_len * config.n_embd * dtype_bytes / 1e9
    activation_memory = activation_per_layer * config.n_layer

    return {
        'model': model_memory,
        'gradients': gradient_memory,
        'optimizer': optimizer_memory,
        'activations': activation_memory,
        'total_training': model_memory + gradient_memory + optimizer_memory + activation_memory,
        'total_inference': model_memory + activation_memory,
    }


def estimate_training_time(config: ModelConfig,
                          dataset_tokens: int,
                          batch_size: int,
                          seq_len: int,
                          hardware_tflops: float,
                          mfu: float = 0.5) -> Dict[str, float]:
    """
    估算MDLM训练时间

    MDLM训练使用加权的MLM损失，计算量与标准BERT/RoBERTa MLM训练相似
    - 不需要多步扩散
    - 每个训练样本只需要一次前向+反向传播

    Args:
        dataset_tokens: 训练数据集的总token数
        hardware_tflops: 硬件的理论峰值性能（TFLOP/s）
        mfu: Model FLOPs Utilization（模型FLOPs利用率，通常0.3-0.6）
    """
    # 计算总的训练步数
    tokens_per_batch = batch_size * seq_len
    total_steps = dataset_tokens // tokens_per_batch

    # 每步的FLOPs（MDLM训练时每个样本只需一次前向+反向）
    flops_per_token = estimate_flops_per_token(config, is_training=True)
    flops_per_step = flops_per_token * tokens_per_batch

    # 总FLOPs
    total_flops = flops_per_step * total_steps

    # 实际吞吐量（考虑MFU）
    actual_tflops = hardware_tflops * mfu

    # 训练时间（秒）
    training_time_seconds = total_flops / (actual_tflops * 1e12)

    return {
        'total_steps': total_steps,
        'total_flops': total_flops,
        'total_tflops': total_flops / 1e12,
        'total_pflops': total_flops / 1e15,
        'training_time_seconds': training_time_seconds,
        'training_time_hours': training_time_seconds / 3600,
        'training_time_days': training_time_seconds / 86400,
        'actual_tflops': actual_tflops,
        'mfu': mfu,
    }


def print_estimation(model_name: str,
                    dataset_tokens: int = 100e9,  # 100B tokens
                    batch_size: int = 256,
                    seq_len: int = 1024,
                    num_gpus: int = 8,
                    gpu_name: str = "A100-80GB",
                    sampling_steps: int = 1000):
    """
    打印MDLM的完整计算资源估算报告
    """
    config = MDLM_CONFIGS[model_name]
    # 覆盖采样步数（如果提供）
    if sampling_steps != config.sampling_steps:
        config.sampling_steps = sampling_steps

    # GPU性能数据（TFLOP/s for BF16/FP16）
    gpu_specs = {
        "A100-80GB": {"tflops": 312, "memory_gb": 80},
        "A100-40GB": {"tflops": 312, "memory_gb": 40},
        "H100-80GB": {"tflops": 989, "memory_gb": 80},
        "V100-32GB": {"tflops": 125, "memory_gb": 32},
        "RTX 4090": {"tflops": 165, "memory_gb": 24},
    }

    gpu_tflops = gpu_specs[gpu_name]["tflops"]
    gpu_memory = gpu_specs[gpu_name]["memory_gb"]

    print("=" * 80)
    print(f"MDLM计算资源估算: {config.name}")
    print("(Masked Diffusion Language Model - MLM as Discrete Diffusion)")
    print("=" * 80)

    # 模型参数
    params = estimate_parameters(config)
    print(f"\n【模型架构 & 参数】")
    print(f"  架构: Encoder-only Transformer (类似BERT/RoBERTa)")
    print(f"  总参数量: {params['total'] / 1e6:.1f}M ({params['total'] / 1e9:.2f}B)")
    print(f"  - Token Embedding: {params['token_embd'] / 1e6:.1f}M")
    print(f"  - Position Embedding: {params['pos_embd'] / 1e6:.1f}M")
    print(f"  - Transformer Layers ({config.n_layer} layers): {params['all_layers'] / 1e6:.1f}M")
    print(f"  - Output Head: {params['output_head'] / 1e6:.1f}M")
    print(f"  注：无需复杂的时间嵌入网络（与连续扩散模型不同）")

    # 内存需求
    print(f"\n【内存需求】")
    mem = estimate_memory(config, batch_size, seq_len)
    print(f"  模型参数: {mem['model']:.2f} GB")
    print(f"  梯度: {mem['gradients']:.2f} GB")
    print(f"  优化器状态: {mem['optimizer']:.2f} GB")
    print(f"  激活值 (batch={batch_size}): {mem['activations']:.2f} GB")
    print(f"  ---")
    print(f"  训练总内存: {mem['total_training']:.2f} GB")
    print(f"  推理总内存: {mem['total_inference']:.2f} GB")

    memory_per_gpu = mem['total_training'] / num_gpus
    print(f"\n  使用 {num_gpus}x {gpu_name}:")
    print(f"    每GPU内存: {memory_per_gpu:.2f} GB")
    if memory_per_gpu > gpu_memory * 0.9:
        print(f"    ⚠️  警告: 超过GPU内存容量 ({gpu_memory} GB)！")
        print(f"    建议: 减小batch size或使用更多GPU")
    else:
        print(f"    ✓ 内存充足 (GPU容量: {gpu_memory} GB)")

    # FLOPs计算
    print(f"\n【FLOPs统计】")
    flops_per_token = estimate_flops_per_token(config, is_training=True)
    print(f"  训练: {flops_per_token / 1e9:.2f} GFLOPs/token")
    print(f"  参数量N: {params['total'] / 1e6:.1f}M")
    print(f"  FLOPs ≈ 6N (前向) + 12N (反向) = {flops_per_token / params['total']:.1f}N")
    print(f"  (与标准BERT/RoBERTa MLM训练相同)")

    # MDLM特有的特性
    print(f"\n【MDLM方法特性】")
    print(f"  训练方法: 加权MLM损失 (覆盖不同masking rates)")
    print(f"  - 每个样本只需1次前向+反向传播")
    print(f"  - 训练效率与标准RoBERTa相同")
    print(f"  推理方法: 迭代demasking")
    print(f"  - 采样步数: {config.sampling_steps} (可调：1000-10000)")
    print(f"  - 每步对整个序列做前向传播")
    print(f"  - 逐步unmask tokens直到生成完整序列")

    # 训练时间估算
    print(f"\n【训练时间估算】")
    print(f"  训练数据: {dataset_tokens / 1e9:.1f}B tokens")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  硬件: {num_gpus}x {gpu_name}")

    total_hardware_tflops = gpu_tflops * num_gpus

    for mfu in [0.3, 0.5, 0.6]:
        timing = estimate_training_time(
            config, dataset_tokens, batch_size, seq_len,
            total_hardware_tflops, mfu
        )
        print(f"\n  MFU = {mfu:.0%} (模型FLOPs利用率):")
        print(f"    总步数: {timing['total_steps']:,}")
        print(f"    总计算量: {timing['total_pflops']:.1f} PFLOPs")
        print(f"    实际算力: {timing['actual_tflops']:.1f} TFLOP/s")
        print(f"    训练时间: {timing['training_time_days']:.1f} 天 ({timing['training_time_hours']:.0f} 小时)")

    # 推理/采样性能
    print(f"\n【采样/生成性能估算】")

    # 测试不同采样步数
    for steps in [1000, 5000, 10000]:
        sampling_flops = estimate_mdlm_sampling_flops(
            config, batch_size=1, seq_len=seq_len,
            num_samples=1, sampling_steps=steps
        )
        single_gpu_time = sampling_flops['total_flops'] / (gpu_tflops * 1e12)
        tokens_per_sec = seq_len / single_gpu_time

        print(f"\n  采样步数 T={steps}:")
        print(f"    生成一个序列 (seq_len={seq_len})")
        print(f"    总FLOPs: {sampling_flops['total_tflops']:.2f} TFLOPs")
        print(f"    在单个{gpu_name}上: {single_gpu_time:.2f} 秒")
        print(f"    生成速度: {tokens_per_sec:.1f} tokens/秒")

    print(f"\n  注意：")
    print(f"    - 采样步数越多，生成质量越好，但速度越慢")
    print(f"    - 可使用更少步数（如1000步）加速生成")
    print(f"    - MDLM比连续扩散模型更高效（离散masking）")

    print("\n" + "=" * 80)


def compare_models():
    """
    比较不同规模的MDLM模型
    """
    print("\n\n")
    print("=" * 80)
    print("MDLM模型规模对比")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Parameters':<15} {'Training Memory':<20} {'Training Time (days)':<25}")
    print("-" * 80)

    for model_name in ["mdlm-base", "mdlm-large", "mdlm-xlarge"]:
        config = MDLM_CONFIGS[model_name]
        params = estimate_parameters(config)
        mem = estimate_memory(config, batch_size=256, seq_len=1024)
        timing = estimate_training_time(
            config,
            dataset_tokens=100e9,
            batch_size=256,
            seq_len=1024,
            hardware_tflops=312 * 8,  # 8x A100
            mfu=0.5
        )

        print(f"{model_name:<20} {params['total']/1e9:>6.2f}B        "
              f"{mem['total_training']:>8.1f} GB          "
              f"{timing['training_time_days']:>10.1f}")

    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("MDLM (Masked Diffusion Language Model) 计算资源估算工具")
    print("基于论文: Simple and Effective Masked Diffusion Language Models (NeurIPS 2024)")
    print("=" * 80)
    print()

    # 估算MDLM-base
    print_estimation(
        model_name="mdlm-base",
        dataset_tokens=100e9,  # 100B tokens (类似MDLM论文在OpenWebText上训练)
        batch_size=256,
        seq_len=1024,
        num_gpus=8,
        gpu_name="A100-80GB",
        sampling_steps=1000
    )

    print("\n\n")

    # 估算MDLM-large
    print_estimation(
        model_name="mdlm-large",
        dataset_tokens=100e9,
        batch_size=128,
        seq_len=1024,
        num_gpus=16,
        gpu_name="A100-80GB",
        sampling_steps=1000
    )

    # 模型对比
    compare_models()

    print("\n")
    print("=" * 80)
    print("关键要点和注意事项:")
    print("=" * 80)
    print("""
【MDLM vs 连续扩散模型】
1. MDLM将MLM视为离散扩散过程，架构更简单（无需复杂时间嵌入）
2. 训练效率与标准BERT/RoBERTa相同（每样本只需1次前向+反向）
3. 推理需要多步迭代demasking，但比连续扩散更高效

【训练】
4. MFU (Model FLOPs Utilization) 通常在30%-60%之间，取决于:
   - 模型大小、Batch size、序列长度
   - 硬件和软件优化水平
5. 训练数据量：MDLM论文使用OpenWebText (~10B tokens)训练1M步

【推理/采样】
6. 采样步数可调（1000-10000步）：
   - 更多步数 → 更高质量，但更慢
   - MDLM论文使用1000步可达到接近AR模型的困惑度（差距15-25%）
7. 加速方法：
   - 减少采样步数（从10000降到1000）
   - 使用缓存优化（ddpm_cache采样器比SEDD快3-4倍）
   - Semi-autoregressive生成

【实现参考】
8. GitHub: https://github.com/kuleshov-group/mdlm
9. 论文: https://arxiv.org/abs/2406.07524
10. ByteDance的Seed Diffusion基于MDLM实现

【估算准确性】
11. 这些估算基于理论计算，实际值可能因实现细节而异
12. 内存估算包含训练时的完整开销，实际可能需要额外buffer
13. FLOPs基于标准Transformer的6N规则（N为参数量）
""")
