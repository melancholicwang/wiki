#!/usr/bin/env python3
"""
估算类RoBERTa的Diffusion Language Model所需的计算资源

参考nanoGPT和nanochat中的估算方法，计算训练和推理的：
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
    """模型配置"""
    name: str
    vocab_size: int = 50265  # RoBERTa vocab size
    max_seq_len: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_ratio: float = 4.0

    # Diffusion特有参数
    diffusion_steps: int = 1000  # 扩散步数
    time_embd_dim: int = 768  # 时间嵌入维度

    def __post_init__(self):
        self.d_head = self.n_embd // self.n_head
        self.d_mlp = int(self.n_embd * self.mlp_ratio)


# 预定义的模型配置
ROBERTA_CONFIGS = {
    "roberta-base": ModelConfig(
        name="roberta-base",
        n_layer=12,
        n_head=12,
        n_embd=768,
    ),
    "roberta-large": ModelConfig(
        name="roberta-large",
        n_layer=24,
        n_head=16,
        n_embd=1024,
    ),
    "roberta-xlarge": ModelConfig(
        name="roberta-xlarge",
        n_layer=36,
        n_head=20,
        n_embd=1280,
    ),
}


def estimate_parameters(config: ModelConfig) -> Dict[str, int]:
    """
    估算模型参数量

    RoBERTa架构包括：
    - Token embedding
    - Position embedding
    - Transformer layers (attention + MLP)
    - Output layer

    Diffusion额外增加：
    - Time embedding network
    - 可能的条件嵌入
    """
    params = {}

    # Token embedding: vocab_size × n_embd
    params['token_embd'] = config.vocab_size * config.n_embd

    # Position embedding: max_seq_len × n_embd
    params['pos_embd'] = config.max_seq_len * config.n_embd

    # Time embedding for diffusion (MLP): diffusion_steps → time_embd_dim → n_embd
    params['time_embd'] = (config.diffusion_steps * config.time_embd_dim +
                           config.time_embd_dim * config.n_embd)

    # 每个Transformer层的参数
    # Self-attention: Q, K, V, O 投影矩阵
    params['attn_qkv_proj_per_layer'] = 3 * config.n_embd * config.n_embd  # Q, K, V
    params['attn_out_proj_per_layer'] = config.n_embd * config.n_embd  # O

    # Layer norm (2 per layer): 2 × (scale + bias)
    params['ln_per_layer'] = 2 * 2 * config.n_embd

    # MLP: two linear layers
    params['mlp_per_layer'] = (config.n_embd * config.d_mlp +  # up projection
                               config.d_mlp * config.n_embd)    # down projection

    # 所有层的总参数
    params_per_layer = (params['attn_qkv_proj_per_layer'] +
                       params['attn_out_proj_per_layer'] +
                       params['ln_per_layer'] +
                       params['mlp_per_layer'])

    params['all_layers'] = params_per_layer * config.n_layer

    # 最终的LayerNorm
    params['final_ln'] = 2 * config.n_embd

    # Output projection (通常与token embedding共享权重，但diffusion可能需要独立的输出层)
    params['output_proj'] = config.n_embd * config.vocab_size

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
    - Diffusion额外的去噪步骤
    """
    params = estimate_parameters(config)
    N = params['total']

    # 基础的Transformer前向传播: 6N per token
    # 包括：2N (matmul) + 4N (attention)
    forward_flops_per_token = 6 * N

    if is_training:
        # 训练时需要前向+反向传播
        # 反向传播约为前向的2倍
        flops_per_token = forward_flops_per_token * 3  # forward + backward
    else:
        flops_per_token = forward_flops_per_token

    return flops_per_token


def estimate_diffusion_flops(config: ModelConfig,
                             batch_size: int,
                             seq_len: int,
                             num_samples: int,
                             is_training: bool = True) -> Dict[str, float]:
    """
    估算Diffusion模型的总FLOPs

    Diffusion模型需要多次前向传播（对应不同的时间步）
    """
    flops_per_token = estimate_flops_per_token(config, is_training)

    # 训练时：每个样本需要采样一个时间步进行去噪
    # 推理时：需要所有diffusion_steps步的去噪
    if is_training:
        num_forward_passes = 1  # 训练时每个样本只采样一个时间步
    else:
        num_forward_passes = config.diffusion_steps  # 推理时需要完整的去噪过程

    total_tokens = batch_size * seq_len * num_samples
    total_flops = flops_per_token * total_tokens * num_forward_passes

    return {
        'total_flops': total_flops,
        'total_tflops': total_flops / 1e12,
        'total_pflops': total_flops / 1e15,
        'flops_per_token': flops_per_token,
        'forward_passes': num_forward_passes,
        'total_tokens': total_tokens,
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
    估算训练时间

    Args:
        dataset_tokens: 训练数据集的总token数
        hardware_tflops: 硬件的理论峰值性能（TFLOP/s）
        mfu: Model FLOPs Utilization（模型FLOPs利用率，通常0.3-0.6）
    """
    # 计算总的训练步数
    tokens_per_batch = batch_size * seq_len
    total_steps = dataset_tokens // tokens_per_batch

    # 每步的FLOPs
    flops_per_step = estimate_diffusion_flops(
        config, batch_size, seq_len, num_samples=1, is_training=True
    )['total_flops']

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
                    seq_len: int = 512,
                    num_gpus: int = 8,
                    gpu_name: str = "A100-80GB"):
    """
    打印完整的估算报告
    """
    config = ROBERTA_CONFIGS[model_name]

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
    print(f"计算资源估算: {config.name} + Diffusion Language Model")
    print("=" * 80)

    # 模型参数
    params = estimate_parameters(config)
    print(f"\n【模型参数】")
    print(f"  总参数量: {params['total'] / 1e6:.1f}M ({params['total'] / 1e9:.2f}B)")
    print(f"  - Token Embedding: {params['token_embd'] / 1e6:.1f}M")
    print(f"  - Position Embedding: {params['pos_embd'] / 1e6:.1f}M")
    print(f"  - Time Embedding (Diffusion): {params['time_embd'] / 1e6:.1f}M")
    print(f"  - Transformer Layers ({config.n_layer} layers): {params['all_layers'] / 1e6:.1f}M")
    print(f"  - Output Projection: {params['output_proj'] / 1e6:.1f}M")

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

    # Diffusion特有的计算
    print(f"\n【Diffusion特性】")
    print(f"  扩散步数: {config.diffusion_steps}")
    print(f"  训练时每样本前向次数: 1 (采样单个时间步)")
    print(f"  推理时每样本前向次数: {config.diffusion_steps} (完整去噪)")

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

    # 推理性能
    print(f"\n【推理性能估算】")
    inference_flops = estimate_diffusion_flops(
        config, batch_size=1, seq_len=seq_len,
        num_samples=1, is_training=False
    )
    print(f"  生成一个序列 (seq_len={seq_len}):")
    print(f"    需要 {config.diffusion_steps} 次去噪步骤")
    print(f"    总FLOPs: {inference_flops['total_tflops']:.2f} TFLOPs")

    # 在单GPU上的推理时间
    single_gpu_time = inference_flops['total_flops'] / (gpu_tflops * 1e12)
    print(f"    在单个{gpu_name}上: {single_gpu_time:.2f} 秒")
    print(f"    生成速度: {seq_len / single_gpu_time:.1f} tokens/秒")

    print("\n" + "=" * 80)


def compare_models():
    """
    比较不同规模的模型
    """
    print("\n\n")
    print("=" * 80)
    print("模型规模对比")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Parameters':<15} {'Training Memory':<20} {'Training Time (days)':<25}")
    print("-" * 80)

    for model_name in ["roberta-base", "roberta-large", "roberta-xlarge"]:
        config = ROBERTA_CONFIGS[model_name]
        params = estimate_parameters(config)
        mem = estimate_memory(config, batch_size=256, seq_len=512)
        timing = estimate_training_time(
            config,
            dataset_tokens=100e9,
            batch_size=256,
            seq_len=512,
            hardware_tflops=312 * 8,  # 8x A100
            mfu=0.5
        )

        print(f"{model_name:<20} {params['total']/1e9:>6.2f}B        "
              f"{mem['total_training']:>8.1f} GB          "
              f"{timing['training_time_days']:>10.1f}")

    print("=" * 80)


if __name__ == "__main__":
    # 估算RoBERTa-base + Diffusion
    print_estimation(
        model_name="roberta-base",
        dataset_tokens=100e9,  # 100B tokens
        batch_size=256,
        seq_len=512,
        num_gpus=8,
        gpu_name="A100-80GB"
    )

    print("\n\n")

    # 估算RoBERTa-large + Diffusion
    print_estimation(
        model_name="roberta-large",
        dataset_tokens=100e9,
        batch_size=128,
        seq_len=512,
        num_gpus=16,
        gpu_name="A100-80GB"
    )

    # 模型对比
    compare_models()

    print("\n")
    print("=" * 80)
    print("注意事项:")
    print("=" * 80)
    print("""
1. 这些估算基于理论计算，实际值可能因实现细节而异
2. MFU (Model FLOPs Utilization) 通常在30%-60%之间，取决于:
   - 模型大小
   - Batch size
   - 序列长度
   - 硬件和软件优化
3. Diffusion模型的推理比标准LM慢很多（需要多步去噪）
4. 可以通过以下方法加速:
   - DDIM采样（减少采样步数）
   - 知识蒸馏
   - 并行去噪
5. 内存估算包含了训练时的完整开销，实际使用可能需要额外的buffer
""")
