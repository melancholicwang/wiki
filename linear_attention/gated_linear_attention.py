"""
Gated Linear Attention (GLA) Implementation

核心逻辑：
GLA 通过移除 softmax 并使用门控机制来实现线性注意力
使用数据依赖的遗忘门（forget gate）来控制信息的保留和遗忘

关键公式：
标准注意力：Attention(Q, K, V) = softmax(QK^T)V
线性注意力：LinearAttn(Q, K, V) = φ(Q)(φ(K)^T V)

GLA 改进：
1. 使用核函数 φ(x) 代替 softmax
2. 添加门控机制：g_t = σ(W_g x_t)
3. 递归形式：S_t = g_t ⊙ S_{t-1} + K_t^T V_t
               O_t = Q_t S_t

复杂度：O(T * D^2) - 线性于序列长度 T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedLinearAttention(nn.Module):
    """
    门控线性注意力

    核心创新：
    1. 线性注意力：避免显式计算注意力矩阵
    2. 门控机制：控制状态的更新和遗忘
    3. 递归形式：通过递归计算实现 O(T) 复杂度
    """

    def __init__(
        self,
        d_model,
        num_heads=8,
        gate_fn="swish",
        use_gate=True,
        use_output_gate=True,
        feature_map="elu",
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_gate = use_gate
        self.use_output_gate = use_output_gate

        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 门控投影
        if use_gate:
            self.w_g = nn.Linear(d_model, d_model, bias=False)  # 遗忘门

        if use_output_gate:
            self.w_o_gate = nn.Linear(d_model, d_model, bias=False)  # 输出门

        # 输出投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 门控激活函数
        if gate_fn == "swish":
            self.gate_fn = lambda x: x * torch.sigmoid(x)
        elif gate_fn == "sigmoid":
            self.gate_fn = torch.sigmoid
        elif gate_fn == "tanh":
            self.gate_fn = torch.tanh
        else:
            self.gate_fn = F.silu

        # 特征映射函数（核函数 φ）
        self.feature_map = feature_map

    def apply_feature_map(self, x):
        """
        应用特征映射 φ(x) 来替代 softmax

        常用的特征映射：
        1. ELU: φ(x) = elu(x) + 1
        2. ReLU: φ(x) = relu(x)
        3. Identity: φ(x) = x
        4. 1 + ELU: 确保正值
        """
        if self.feature_map == "elu":
            return F.elu(x) + 1  # 确保输出 > 0
        elif self.feature_map == "relu":
            return F.relu(x)
        elif self.feature_map == "identity":
            return x
        elif self.feature_map == "1+elu":
            return 1 + F.elu(x)
        else:
            return F.elu(x) + 1

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # 2. 应用特征映射（代替 softmax）
        Q = self.apply_feature_map(Q)
        K = self.apply_feature_map(K)

        # 3. 计算门控值（如果启用）
        if self.use_gate:
            # 遗忘门：控制信息的保留程度
            G = self.w_g(x).view(batch_size, seq_len, self.num_heads, self.d_k)
            G = self.gate_fn(G)  # (B, L, H, D_k)
        else:
            G = None

        # 4. 门控线性注意力递归计算
        output = self.gla_recurrence(Q, K, V, G)

        # 5. 输出门控（如果启用）
        if self.use_output_gate:
            O_gate = self.w_o_gate(x)
            O_gate = self.gate_fn(O_gate)
            output = output * O_gate

        # 6. 输出投影
        output = self.w_o(output)

        return output

    def gla_recurrence(self, Q, K, V, G=None):
        """
        GLA 递归计算

        递归公式：
        如果有门控：S_t = G_t ⊙ S_{t-1} + K_t^T V_t
        否则：S_t = S_{t-1} + K_t^T V_t
        输出：O_t = Q_t S_t

        这是线性注意力的关键优化：
        - 不计算完整的 QK^T 矩阵（T×T）
        - 而是维护一个累积状态 S（D×D）
        - 复杂度从 O(T^2 D) 降到 O(T D^2)

        Args:
            Q: (B, L, H, D_k) - 查询
            K: (B, L, H, D_k) - 键
            V: (B, L, H, D_k) - 值
            G: (B, L, H, D_k) or None - 门控

        Returns:
            output: (B, L, D) - 输出
        """
        batch_size, seq_len, num_heads, d_k = Q.shape

        # 初始化状态矩阵 S: (B, H, D_k, D_k)
        # S 代表累积的 K^T V 外积
        S = torch.zeros(
            batch_size, num_heads, d_k, d_k,
            device=Q.device, dtype=Q.dtype
        )

        # 归一化项（可选，用于数值稳定性）
        normalizer = torch.zeros(
            batch_size, num_heads, d_k,
            device=Q.device, dtype=Q.dtype
        )

        outputs = []

        for t in range(seq_len):
            # 当前时刻的 Q, K, V
            q_t = Q[:, t]  # (B, H, D_k)
            k_t = K[:, t]  # (B, H, D_k)
            v_t = V[:, t]  # (B, H, D_k)

            # 应用门控（遗忘机制）
            if G is not None:
                g_t = G[:, t]  # (B, H, D_k)
                # 门控应用到状态
                # 类似于 LSTM 的遗忘门：f_t ⊙ h_{t-1}
                S = g_t.unsqueeze(-1) * S  # (B, H, D_k, D_k)
                normalizer = g_t * normalizer  # (B, H, D_k)

            # 更新状态：S_t = S_{t-1} + K_t^T V_t
            # K_t^T V_t 是外积：(D_k, 1) @ (1, D_k) = (D_k, D_k)
            kv = k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)  # (B, H, D_k, D_k)
            S = S + kv

            # 更新归一化项
            normalizer = normalizer + k_t  # (B, H, D_k)

            # 计算输出：O_t = Q_t @ S_t
            # (B, H, D_k) @ (B, H, D_k, D_k) -> (B, H, D_k)
            o_t = torch.einsum('bhd,bhde->bhe', q_t, S)

            # 归一化（防止数值爆炸）
            # o_t = o_t / (q_t @ normalizer + 1e-6)
            norm = torch.einsum('bhd,bhd->bh', q_t, normalizer).unsqueeze(-1) + 1e-6
            o_t = o_t / norm

            outputs.append(o_t)

        # 合并所有时刻的输出
        output = torch.stack(outputs, dim=1)  # (B, L, H, D_k)
        output = output.reshape(batch_size, seq_len, self.d_model)

        return output


class ParallelGLA(nn.Module):
    """
    并行 GLA - 训练时使用

    虽然 GLA 可以递归计算，但训练时可以使用并行算法加速
    使用累积和（cumsum）和并行扫描算法
    """

    def __init__(self, d_model, num_heads=8, feature_map="elu"):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.feature_map = feature_map

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def apply_feature_map(self, x):
        if self.feature_map == "elu":
            return F.elu(x) + 1
        else:
            return F.elu(x) + 1

    def forward(self, x):
        """
        并行计算版本 - 适合训练

        使用矩阵运算并行计算所有时间步
        """
        batch_size, seq_len, _ = x.shape

        # 投影
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        G = self.gate_fn(self.w_g(x)).view(batch_size, seq_len, self.num_heads, self.d_k)

        Q = self.apply_feature_map(Q)
        K = self.apply_feature_map(K)

        # 转置以便并行计算
        # (B, L, H, D) -> (B, H, L, D)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        G = G.transpose(1, 2)

        # 计算累积门控（使用 log 空间避免下溢）
        log_g = torch.log(G.clamp(min=1e-6))
        log_g_cumsum = torch.cumsum(log_g, dim=2)  # (B, H, L, D)

        # 并行计算注意力
        # 这部分可以使用高效的并行扫描算法（associative scan）
        # 简化实现使用循环（实际应用中应使用优化版本）
        output = []
        for h in range(self.num_heads):
            o_h = self._parallel_scan_head(
                Q[:, h], K[:, h], V[:, h], log_g_cumsum[:, h]
            )
            output.append(o_h)

        output = torch.stack(output, dim=1)  # (B, H, L, D)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        return output

    def _parallel_scan_head(self, Q, K, V, log_g_cumsum):
        """单个头的并行扫描"""
        batch_size, seq_len, d_k = Q.shape

        outputs = []
        for b in range(batch_size):
            S = torch.zeros(d_k, d_k, device=Q.device, dtype=Q.dtype)
            batch_out = []

            for t in range(seq_len):
                # 应用累积门控
                decay = torch.exp(log_g_cumsum[b, t])
                S = decay.unsqueeze(-1) * S

                # 更新状态
                kv = K[b, t].unsqueeze(-1) @ V[b, t].unsqueeze(0)
                S = S + kv

                # 计算输出
                o_t = Q[b, t] @ S
                batch_out.append(o_t)

            outputs.append(torch.stack(batch_out))

        return torch.stack(outputs)

    def gate_fn(self, x):
        return x * torch.sigmoid(x)


# 技术分析示例
if __name__ == "__main__":
    print("=" * 80)
    print("Gated Linear Attention (GLA) 技术分析")
    print("=" * 80)

    # 创建示例
    batch_size, seq_len, d_model = 2, 100, 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    # GLA
    gla = GatedLinearAttention(d_model, num_heads)
    output = gla(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 计算复杂度分析
    d_k = d_model // num_heads
    print("\n计算复杂度分析:")
    print(f"GLA 复杂度: O(T * D^2) = O({seq_len} * {d_model}^2)")
    print(f"  = {seq_len * d_model * d_model} 操作")
    print(f"标准注意力: O(T^2 * D) = O({seq_len}^2 * {d_model})")
    print(f"  = {seq_len**2 * d_model} 操作")

    print(f"\n当 T > D 时，GLA 更快：")
    print(f"  T={seq_len}, D={d_model}: GLA 快 {(seq_len**2 * d_model) / (seq_len * d_model * d_model):.2f}x")
    print(f"  T=1024, D={d_model}: GLA 快 {(1024**2 * d_model) / (1024 * d_model * d_model):.2f}x")

    # 内存使用
    state_size = batch_size * num_heads * d_k * d_k * 4  # float32
    print(f"\n内存使用:")
    print(f"状态矩阵 S: {state_size / 1024:.2f} KB")
    print(f"  vs. 注意力矩阵: {batch_size * num_heads * seq_len * seq_len * 4 / 1024:.2f} KB")

    print("\n核心创新:")
    print("1. 线性注意力:")
    print("   - 不计算显式的 T×T 注意力矩阵")
    print("   - 使用核技巧：φ(Q)(φ(K)^T V)")
    print("   - 维护 D×D 状态矩阵而非 T×T 矩阵")
    print("\n2. 门控机制:")
    print("   - 数据依赖的遗忘门")
    print("   - 类似 LSTM，但更简单高效")
    print("   - 控制历史信息的保留和遗忘")
    print("\n3. 递归形式:")
    print("   - S_t = g_t ⊙ S_{t-1} + K_t^T V_t")
    print("   - O_t = Q_t S_t")
    print("   - 推理时 O(1) 时间复杂度")

    print("\n优缺点:")
    print("优点:")
    print("  - O(T D^2) 复杂度，长序列时远快于标准注意力")
    print("  - 推理时只需常数时间（递归状态）")
    print("  - 内存效率高")
    print("  - 可以处理长序列")
    print("\n缺点:")
    print("  - 表达能力弱于 softmax 注意力")
    print("  - 门控需要仔细调优")
    print("  - 特征映射的选择影响性能")
    print("  - 训练时需要特殊的并行算法")

    print("\n适用场景:")
    print("  - 长序列建模（文档、代码）")
    print("  - 需要快速推理的应用")
    print("  - 流式处理场景")
    print("  - 资源受限的环境")
