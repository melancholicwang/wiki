"""
Standard Attention (SA) Implementation

核心逻辑：
标准的自注意力机制使用 softmax 归一化，时间复杂度为 O(T^2 * D)
公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

特点：
1. 使用 softmax 进行归一化
2. 二次复杂度 O(T^2)
3. 能够捕获全局依赖关系
4. 需要存储完整的注意力矩阵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StandardAttention(nn.Module):
    """标准的多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) 可选的掩码

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影并分割为多头
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数
        # (batch_size, num_heads, seq_len, seq_len)
        # 核心计算：QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax 归一化（关键步骤：确保注意力权重和为1）
        # 这一步的复杂度是 O(T^2)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 5. 应用注意力权重到 V
        # (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attention_weights, V)

        # 6. 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 7. 输出投影
        output = self.w_o(context)

        return output, attention_weights


class CausalStandardAttention(StandardAttention):
    """因果（Causal）标准注意力 - 用于自回归模型"""

    def forward(self, x):
        """
        自动应用因果掩码，确保位置 i 只能关注位置 j <= i
        """
        batch_size, seq_len, _ = x.shape

        # 创建因果掩码
        # 下三角矩阵，允许当前位置看到之前的位置
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        return super().forward(x, mask=causal_mask)


# 技术分析示例
if __name__ == "__main__":
    print("=" * 80)
    print("Standard Attention 技术分析")
    print("=" * 80)

    # 创建示例
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    # 标准注意力
    sa = StandardAttention(d_model, num_heads)
    output, attn_weights = sa(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")

    # 分析计算复杂度
    print("\n计算复杂度分析:")
    print(f"QK^T 矩阵乘法: O(T^2 * D) = O({seq_len}^2 * {d_model}) = {seq_len**2 * d_model} FLOPs")
    print(f"Softmax: O(T^2) = O({seq_len}^2) = {seq_len**2} 操作")
    print(f"注意力 * V: O(T^2 * D) = O({seq_len}^2 * {d_model}) = {seq_len**2 * d_model} FLOPs")
    print(f"总复杂度: O(T^2 * D)")

    # 内存使用
    attn_matrix_size = batch_size * num_heads * seq_len * seq_len * 4  # float32
    print(f"\n内存使用:")
    print(f"注意力矩阵大小: {attn_matrix_size / 1024:.2f} KB")
    print(f"对于 seq_len=1024: {batch_size * num_heads * 1024 * 1024 * 4 / 1024 / 1024:.2f} MB")

    print("\n优缺点:")
    print("优点:")
    print("  - 能够捕获任意距离的依赖关系")
    print("  - 理论上具有最强的表达能力")
    print("  - 注意力权重可解释性强")
    print("\n缺点:")
    print("  - O(T^2) 复杂度，长序列时非常慢")
    print("  - 需要存储完整的 T×T 注意力矩阵，内存消耗大")
    print("  - 无法处理超长序列（如 100K tokens）")
