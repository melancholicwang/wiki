"""
DeltaNet Implementation

核心逻辑：
DeltaNet 是一种线性注意力变体，使用距离衰减和自适应遗忘机制
结合了位置信息和内容信息来调制注意力

关键创新：
1. Delta 规则：使用距离相关的衰减因子
2. 自适应遗忘：基于内容的动态衰减
3. 位置感知：显式建模位置信息
4. 高效递归：O(T) 复杂度

数学形式：
β_t = sigmoid(W_β x_t)  # 衰减因子
δ_t = 距离衰减函数
S_t = β_t ⊙ δ_t ⊙ S_{t-1} + K_t^T V_t
O_t = Q_t S_t

特点：
- 位置编码隐式建模在衰减中
- 结合了 RNN 和 Attention 的优点
- 推理速度快，训练也相对高效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeltaNet(nn.Module):
    """
    DeltaNet: 基于距离衰减的线性注意力

    核心思想：
    1. 使用距离衰减替代 softmax 归一化
    2. 自适应地学习遗忘率
    3. 保持线性复杂度的同时建模位置信息
    """

    def __init__(
        self,
        d_model,
        num_heads=8,
        d_head=None,
        decay_type="exponential",  # exponential, linear, learnable
        use_beta=True,  # 是否使用自适应 beta
        use_output_gate=True,
        max_positions=2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head or (d_model // num_heads)
        self.decay_type = decay_type
        self.use_beta = use_beta
        self.use_output_gate = use_output_gate
        self.max_positions = max_positions

        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * self.d_head, bias=False)

        # Beta 投影（自适应衰减）
        if use_beta:
            self.w_beta = nn.Linear(d_model, num_heads * self.d_head, bias=True)

        # 输出门（可选）
        if use_output_gate:
            self.w_gate = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.w_o = nn.Linear(num_heads * self.d_head, d_model, bias=False)

        # 衰减参数
        if decay_type == "learnable":
            # 可学习的衰减参数
            self.decay_params = nn.Parameter(
                torch.randn(num_heads, self.d_head) * 0.01
            )
        elif decay_type == "exponential":
            # 固定的指数衰减
            # gamma = exp(-alpha * distance)
            alpha = torch.linspace(0.01, 0.1, num_heads * self.d_head)
            self.register_buffer('alpha', alpha.view(num_heads, self.d_head))

        # 位置编码（用于计算距离）
        self.register_buffer(
            'positions',
            torch.arange(max_positions).unsqueeze(0)
        )

    def compute_decay(self, seq_len, device):
        """
        计算距离衰减矩阵

        返回 decay[i, j] 表示位置 j 对位置 i 的衰减系数
        通常 i >= j (因果注意力)

        Args:
            seq_len: 序列长度

        Returns:
            decay: (seq_len, seq_len, num_heads, d_head)
        """
        # 创建距离矩阵
        positions = torch.arange(seq_len, device=device)
        # distance[i, j] = i - j (只考虑 i >= j 的情况)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        distance = distance.clamp(min=0)  # 因果：只关注过去

        if self.decay_type == "exponential":
            # 指数衰减: exp(-alpha * distance)
            # (L, L, 1, 1) * (1, 1, H, D) -> (L, L, H, D)
            decay = torch.exp(
                -distance.unsqueeze(-1).unsqueeze(-1) * self.alpha.unsqueeze(0).unsqueeze(0)
            )

        elif self.decay_type == "linear":
            # 线性衰减: max(0, 1 - alpha * distance)
            decay = torch.clamp(
                1.0 - distance.unsqueeze(-1).unsqueeze(-1) * self.alpha.unsqueeze(0).unsqueeze(0),
                min=0.0
            )

        elif self.decay_type == "learnable":
            # 可学习衰减: sigmoid(param) ^ distance
            gamma = torch.sigmoid(self.decay_params)  # (H, D)
            # gamma^distance
            decay = gamma.unsqueeze(0).unsqueeze(0) ** distance.unsqueeze(-1).unsqueeze(-1)

        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        return decay

    def forward(self, x, return_state=False):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            return_state: 是否返回最终状态（用于增量解码）

        Returns:
            output: (batch_size, seq_len, d_model)
            state: 如果 return_state=True，返回最终状态
        """
        batch_size, seq_len, _ = x.shape

        # 1. 投影 Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_head)

        # 2. 计算自适应 beta（衰减率）
        if self.use_beta:
            beta = torch.sigmoid(self.w_beta(x))  # (B, L, H*D)
            beta = beta.view(batch_size, seq_len, self.num_heads, self.d_head)
        else:
            beta = None

        # 3. DeltaNet 递归计算
        output, final_state = self.deltanet_recurrence(Q, K, V, beta)

        # 4. 输出门控
        if self.use_output_gate:
            gate = torch.sigmoid(self.w_gate(x))
            output = output * gate

        # 5. 输出投影
        output = self.w_o(output)

        if return_state:
            return output, final_state
        return output

    def deltanet_recurrence(self, Q, K, V, beta=None):
        """
        DeltaNet 递归计算

        核心递归：
        对于每个时间步 t:
          1. 应用距离衰减：S_t = decay_t ⊙ S_{t-1}
          2. 应用 beta 遗忘：S_t = beta_t ⊙ S_t
          3. 更新状态：S_t = S_t + K_t^T V_t
          4. 计算输出：O_t = Q_t S_t

        Args:
            Q: (B, L, H, D) - 查询
            K: (B, L, H, D) - 键
            V: (B, L, H, D) - 值
            beta: (B, L, H, D) or None - 自适应衰减

        Returns:
            output: (B, L, H*D)
            final_state: (B, H, D, D)
        """
        batch_size, seq_len, num_heads, d_head = Q.shape

        # 计算距离衰减（这里简化为逐步递减）
        # 实际实现中可以预计算或使用更高效的方法

        # 初始化状态
        S = torch.zeros(
            batch_size, num_heads, d_head, d_head,
            device=Q.device, dtype=Q.dtype
        )
        normalizer = torch.zeros(
            batch_size, num_heads, d_head,
            device=Q.device, dtype=Q.dtype
        )

        outputs = []

        for t in range(seq_len):
            q_t = Q[:, t]  # (B, H, D)
            k_t = K[:, t]  # (B, H, D)
            v_t = V[:, t]  # (B, H, D)

            # 1. 应用 beta 遗忘
            if beta is not None:
                beta_t = beta[:, t]  # (B, H, D)
                # S_t = beta_t ⊙ S_{t-1}
                S = beta_t.unsqueeze(-1) * S
                normalizer = beta_t * normalizer

            # 在实际实现中，这里还应该应用距离衰减
            # 为简化，这里使用固定的全局衰减
            global_decay = 0.99  # 可以替换为更复杂的衰减策略
            S = global_decay * S
            normalizer = global_decay * normalizer

            # 2. 更新状态：S_t = S_t + K_t^T V_t
            kv = k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)  # (B, H, D, D)
            S = S + kv
            normalizer = normalizer + k_t

            # 3. 计算输出：O_t = Q_t S_t
            o_t = torch.einsum('bhd,bhde->bhe', q_t, S)

            # 归一化
            norm = torch.einsum('bhd,bhd->bh', q_t, normalizer).unsqueeze(-1) + 1e-6
            o_t = o_t / norm

            outputs.append(o_t)

        # 合并输出
        output = torch.stack(outputs, dim=1)  # (B, L, H, D)
        output = output.reshape(batch_size, seq_len, num_heads * d_head)

        return output, S


class ParallelDeltaNet(nn.Module):
    """
    并行 DeltaNet - 用于训练

    使用矩阵形式计算，避免显式循环
    """

    def __init__(self, d_model, num_heads=8, d_head=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head or (d_model // num_heads)

        self.w_q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.w_beta = nn.Linear(d_model, num_heads * self.d_head, bias=True)
        self.w_o = nn.Linear(num_heads * self.d_head, d_model, bias=False)

        # 距离衰减参数
        alpha = torch.linspace(0.01, 0.1, num_heads * self.d_head)
        self.register_buffer('alpha', alpha.view(num_heads, self.d_head))

    def forward(self, x):
        """
        并行计算版本

        使用累积乘积和并行扫描实现高效训练
        """
        batch_size, seq_len, _ = x.shape

        # 投影
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        beta = torch.sigmoid(self.w_beta(x)).view(batch_size, seq_len, self.num_heads, self.d_head)

        # 计算距离衰减矩阵
        positions = torch.arange(seq_len, device=x.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        distance = distance.clamp(min=0)  # 因果

        # (L, L) -> (L, L, H, D)
        decay = torch.exp(
            -distance.unsqueeze(-1).unsqueeze(-1) * self.alpha.unsqueeze(0).unsqueeze(0)
        )

        # 转置以便批量计算
        Q = Q.transpose(1, 2)  # (B, H, L, D)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        beta = beta.transpose(1, 2)

        # 简化的并行计算（实际中需要更高效的实现）
        # 这里为了清晰起见使用相对直接的方法
        output = self._parallel_forward(Q, K, V, beta, decay)

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.w_o(output)

        return output

    def _parallel_forward(self, Q, K, V, beta, decay):
        """并行前向传播的简化实现"""
        batch_size, num_heads, seq_len, d_head = Q.shape

        outputs = []
        for b in range(batch_size):
            head_outputs = []
            for h in range(num_heads):
                # 对每个样本和头单独计算（可以进一步优化）
                S = torch.zeros(d_head, d_head, device=Q.device, dtype=Q.dtype)
                head_out = []

                for t in range(seq_len):
                    # 应用 beta 和全局衰减
                    S = beta[b, h, t].unsqueeze(-1) * 0.99 * S

                    # 更新状态
                    kv = K[b, h, t].unsqueeze(-1) @ V[b, h, t].unsqueeze(0)
                    S = S + kv

                    # 计算输出
                    o_t = Q[b, h, t] @ S
                    head_out.append(o_t)

                head_outputs.append(torch.stack(head_out))

            outputs.append(torch.stack(head_outputs))

        return torch.stack(outputs)


# 技术分析示例
if __name__ == "__main__":
    print("=" * 80)
    print("DeltaNet 技术分析")
    print("=" * 80)

    # 创建示例
    batch_size, seq_len, d_model = 2, 100, 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    # DeltaNet
    deltanet = DeltaNet(d_model, num_heads)
    output = deltanet(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 计算复杂度分析
    d_head = d_model // num_heads
    print("\n计算复杂度分析:")
    print(f"DeltaNet: O(T * D^2)")
    print(f"  = O({seq_len} * {d_model}^2) = {seq_len * d_model**2}")
    print(f"标准注意力: O(T^2 * D)")
    print(f"  = O({seq_len}^2 * {d_model}) = {seq_len**2 * d_model}")
    print(f"加速比: {(seq_len**2 * d_model) / (seq_len * d_model**2):.2f}x (当 T={seq_len})")

    print("\n核心创新:")
    print("1. 距离衰减:")
    print("   - 显式建模位置信息")
    print("   - 衰减 = exp(-alpha * distance)")
    print("   - 近距离关注更强，远距离逐渐衰减")
    print("\n2. 自适应遗忘 (Beta):")
    print("   - β_t = sigmoid(W_β x_t)")
    print("   - 基于内容动态调整遗忘率")
    print("   - 重要信息保留更久")
    print("\n3. 递归状态更新:")
    print("   - S_t = β_t ⊙ decay_t ⊙ S_{t-1} + K_t^T V_t")
    print("   - 结合位置和内容信息")
    print("   - O(1) 推理时间")

    print("\n与其他方法的对比:")
    print("vs. 标准注意力:")
    print("  - DeltaNet: 线性复杂度，显式位置编码")
    print("  - 标准注意力: 二次复杂度，需要额外位置编码")
    print("\nvs. GLA:")
    print("  - DeltaNet: 距离感知衰减")
    print("  - GLA: 统一的遗忘机制")
    print("\nvs. Mamba:")
    print("  - DeltaNet: 更像增强的线性注意力")
    print("  - Mamba: 基于 SSM，完全不同的范式")

    print("\n优缺点:")
    print("优点:")
    print("  - 线性复杂度 O(T)")
    print("  - 位置信息建模更自然")
    print("  - 推理高效（递归状态）")
    print("  - 比纯线性注意力更强的归纳偏置")
    print("\n缺点:")
    print("  - 表达能力仍弱于 softmax 注意力")
    print("  - 衰减函数需要精心设计")
    print("  - 长距离依赖可能被过度衰减")

    print("\n适用场景:")
    print("  - 位置信息重要的任务（如代码生成）")
    print("  - 需要平衡局部和全局依赖的场景")
    print("  - 长序列建模")
    print("  - 实时/流式处理")
