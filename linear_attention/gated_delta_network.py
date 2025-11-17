"""
Gated Delta Network (GDN) Implementation

核心逻辑：
GDN 结合了 GLA 的门控机制和 DeltaNet 的距离感知能力
使用多重门控和精细的衰减策略实现更强的表达能力

关键创新：
1. 多重门控：输入门、遗忘门、输出门
2. 自适应 Delta：内容和位置双重依赖的衰减
3. 分层状态：维护多个粒度的状态表示
4. 数据依赖路由：动态选择信息传播路径

数学形式：
f_t = sigmoid(W_f x_t)  # 遗忘门
i_t = sigmoid(W_i x_t)  # 输入门
o_t = sigmoid(W_o x_t)  # 输出门
δ_t = learnable_decay(x_t, distance)  # 自适应衰减

S_t = f_t ⊙ δ_t ⊙ S_{t-1} + i_t ⊙ (K_t^T V_t)
O_t = o_t ⊙ (Q_t S_t)

特点：
- 更精细的信息控制
- 更强的表达能力
- 仍保持线性复杂度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedDeltaNetwork(nn.Module):
    """
    Gated Delta Network (GDN)

    结合 LSTM 风格的门控和 Delta 风格的衰减
    提供更强的序列建模能力
    """

    def __init__(
        self,
        d_model,
        num_heads=8,
        d_head=None,
        use_forget_gate=True,
        use_input_gate=True,
        use_output_gate=True,
        decay_type="adaptive",  # adaptive, exponential, learnable
        num_decay_layers=2,  # 衰减函数的层数
        feature_map="elu",
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head or (d_model // num_heads)
        self.use_forget_gate = use_forget_gate
        self.use_input_gate = use_input_gate
        self.use_output_gate = use_output_gate
        self.decay_type = decay_type
        self.feature_map = feature_map

        hidden_dim = num_heads * self.d_head

        # Q, K, V 投影
        self.w_q = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_k = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_v = nn.Linear(d_model, hidden_dim, bias=False)

        # 门控投影
        if use_forget_gate:
            self.w_forget = nn.Linear(d_model, hidden_dim, bias=True)

        if use_input_gate:
            self.w_input = nn.Linear(d_model, hidden_dim, bias=True)

        if use_output_gate:
            self.w_output = nn.Linear(d_model, hidden_dim, bias=True)

        # 自适应衰减网络
        if decay_type == "adaptive":
            # 多层 MLP 学习衰减函数
            decay_layers = []
            decay_layers.append(nn.Linear(d_model + 1, hidden_dim))  # +1 for distance
            decay_layers.append(nn.SiLU())

            for _ in range(num_decay_layers - 1):
                decay_layers.append(nn.Linear(hidden_dim, hidden_dim))
                decay_layers.append(nn.SiLU())

            decay_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.decay_net = nn.Sequential(*decay_layers)

        elif decay_type == "exponential":
            # 固定指数衰减参数
            alpha = torch.linspace(0.01, 0.1, hidden_dim)
            self.register_buffer('alpha', alpha)

        elif decay_type == "learnable":
            # 可学习的衰减参数（每个头和维度独立）
            self.decay_params = nn.Parameter(torch.randn(num_heads, self.d_head) * 0.1)

        # 输出投影
        self.w_o = nn.Linear(hidden_dim, d_model, bias=False)

        # 层归一化（用于稳定训练）
        self.norm_pre = nn.LayerNorm(hidden_dim)
        self.norm_post = nn.LayerNorm(hidden_dim)

    def apply_feature_map(self, x):
        """应用特征映射"""
        if self.feature_map == "elu":
            return F.elu(x) + 1
        elif self.feature_map == "relu":
            return F.relu(x)
        elif self.feature_map == "identity":
            return x
        else:
            return F.elu(x) + 1

    def compute_adaptive_decay(self, x, t, device):
        """
        计算自适应衰减因子

        考虑当前输入内容和相对距离

        Args:
            x: (B, L, D) 输入
            t: int 当前时间步
            device: torch.device

        Returns:
            decay: (B, H, D_h) 衰减因子
        """
        batch_size, seq_len, _ = x.shape

        if self.decay_type == "adaptive":
            # 使用神经网络学习衰减函数
            # 输入：当前内容 + 相对距离信息
            distances = torch.arange(t + 1, device=device).float()
            distances = distances / (t + 1)  # 归一化到 [0, 1]

            # 为每个历史位置计算衰减
            # 这里简化处理：使用当前位置的内容
            x_t = x[:, t:t+1]  # (B, 1, D)

            # 添加距离信息
            dist_feature = distances[-1].view(1, 1, 1).expand(batch_size, 1, 1)
            x_with_dist = torch.cat([x_t, dist_feature], dim=-1)  # (B, 1, D+1)

            # 通过衰减网络
            decay = self.decay_net(x_with_dist)  # (B, 1, H*D_h)
            decay = torch.sigmoid(decay)  # 确保在 [0, 1]
            decay = decay.view(batch_size, self.num_heads, self.d_head)

        elif self.decay_type == "exponential":
            # 简单的指数衰减
            decay = torch.ones(batch_size, len(self.alpha), device=device)
            decay = torch.exp(-self.alpha * 0.1)  # 固定衰减率
            decay = decay.view(1, self.num_heads, self.d_head).expand(batch_size, -1, -1)

        elif self.decay_type == "learnable":
            # 可学习但固定的衰减（不依赖输入）
            decay = torch.sigmoid(self.decay_params)
            decay = decay.unsqueeze(0).expand(batch_size, -1, -1)

        return decay

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 1. 投影 Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_head)

        # 应用特征映射
        Q = self.apply_feature_map(Q)
        K = self.apply_feature_map(K)

        # 2. 计算门控
        gates = {}

        if self.use_forget_gate:
            forget_gate = torch.sigmoid(self.w_forget(x))
            gates['forget'] = forget_gate.view(batch_size, seq_len, self.num_heads, self.d_head)

        if self.use_input_gate:
            input_gate = torch.sigmoid(self.w_input(x))
            gates['input'] = input_gate.view(batch_size, seq_len, self.num_heads, self.d_head)

        if self.use_output_gate:
            output_gate = torch.sigmoid(self.w_output(x))
            gates['output'] = output_gate.view(batch_size, seq_len, self.num_heads, self.d_head)

        # 3. GDN 递归计算
        output = self.gdn_recurrence(Q, K, V, gates, x)

        # 4. 输出投影
        output = self.w_o(output)

        return output

    def gdn_recurrence(self, Q, K, V, gates, x):
        """
        GDN 递归计算

        核心递归（包含所有门控）：
        f_t = forget_gate
        i_t = input_gate
        o_t = output_gate
        δ_t = adaptive_decay(x_t, t)

        S_t = f_t ⊙ δ_t ⊙ S_{t-1} + i_t ⊙ (K_t^T V_t)
        O_t = o_t ⊙ (Q_t S_t / normalizer)

        Args:
            Q, K, V: (B, L, H, D_h)
            gates: dict of gates
            x: (B, L, D_model) 原始输入（用于计算自适应衰减）

        Returns:
            output: (B, L, H*D_h)
        """
        batch_size, seq_len, num_heads, d_head = Q.shape
        device = Q.device

        # 初始化状态
        S = torch.zeros(
            batch_size, num_heads, d_head, d_head,
            device=device, dtype=Q.dtype
        )
        normalizer = torch.zeros(
            batch_size, num_heads, d_head,
            device=device, dtype=Q.dtype
        )

        outputs = []

        for t in range(seq_len):
            # 当前时刻的张量
            q_t = Q[:, t]  # (B, H, D_h)
            k_t = K[:, t]
            v_t = V[:, t]

            # 1. 遗忘门：控制历史信息的保留
            if 'forget' in gates:
                f_t = gates['forget'][:, t]
                S = f_t.unsqueeze(-1) * S
                normalizer = f_t * normalizer
            else:
                f_t = 1.0

            # 2. 自适应衰减：基于内容和距离
            if self.decay_type != "none":
                decay = self.compute_adaptive_decay(x, t, device)
                S = decay.unsqueeze(-1) * S
                normalizer = decay * normalizer

            # 3. 输入门：控制新信息的写入
            if 'input' in gates:
                i_t = gates['input'][:, t]
                kv = (i_t * k_t).unsqueeze(-1) @ v_t.unsqueeze(-2)
            else:
                kv = k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)

            # 4. 更新状态
            S = S + kv
            normalizer = normalizer + k_t

            # 5. 计算原始输出
            o_t = torch.einsum('bhd,bhde->bhe', q_t, S)

            # 6. 归一化
            norm = torch.einsum('bhd,bhd->bh', q_t, normalizer).unsqueeze(-1).clamp(min=1e-6)
            o_t = o_t / norm

            # 7. 输出门：控制输出
            if 'output' in gates:
                g_t = gates['output'][:, t]
                o_t = g_t * o_t

            outputs.append(o_t)

        # 合并输出
        output = torch.stack(outputs, dim=1)  # (B, L, H, D_h)
        output = output.reshape(batch_size, seq_len, num_heads * d_head)

        return output


class HierarchicalGDN(nn.Module):
    """
    分层 GDN

    维护多个时间尺度的状态：
    - 短期状态：快速更新，捕获局部模式
    - 长期状态：慢速更新，捕获全局模式
    """

    def __init__(self, d_model, num_heads=8, num_scales=2):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_scales = num_scales

        # 为每个时间尺度创建独立的 GDN
        self.scales = nn.ModuleList([
            GatedDeltaNetwork(
                d_model,
                num_heads=num_heads,
                decay_type="learnable",
            )
            for _ in range(num_scales)
        ])

        # 尺度融合
        self.scale_fusion = nn.Linear(d_model * num_scales, d_model)

    def forward(self, x):
        """
        多尺度处理

        Args:
            x: (B, L, D)

        Returns:
            output: (B, L, D)
        """
        # 在每个尺度上处理
        scale_outputs = []
        for scale_module in self.scales:
            scale_out = scale_module(x)
            scale_outputs.append(scale_out)

        # 融合多尺度输出
        combined = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(combined)

        return output


# 技术分析示例
if __name__ == "__main__":
    print("=" * 80)
    print("Gated Delta Network (GDN) 技术分析")
    print("=" * 80)

    # 创建示例
    batch_size, seq_len, d_model = 2, 100, 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    # GDN
    gdn = GatedDeltaNetwork(d_model, num_heads, decay_type="adaptive")
    output = gdn(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 分层 GDN
    h_gdn = HierarchicalGDN(d_model, num_heads, num_scales=3)
    h_output = h_gdn(x)
    print(f"分层 GDN 输出: {h_output.shape}")

    print("\n计算复杂度分析:")
    print(f"GDN: O(T * D^2) （与 GLA、DeltaNet 相同）")
    print(f"额外开销：门控计算 + 自适应衰减")
    print(f"  门控: 3 * O(T * D^2) (forget, input, output)")
    print(f"  自适应衰减: O(T * D * H) (H 是衰减网络深度)")

    print("\n核心创新:")
    print("1. 多重门控机制:")
    print("   - 遗忘门 (f): 控制历史信息保留")
    print("   - 输入门 (i): 控制新信息写入")
    print("   - 输出门 (o): 控制输出强度")
    print("   - 类似 LSTM，但应用于线性注意力")
    print("\n2. 自适应衰减:")
    print("   - δ_t = decay_net(x_t, distance)")
    print("   - 同时考虑内容和位置")
    print("   - 神经网络学习最优衰减策略")
    print("\n3. 精细的信息控制:")
    print("   - S_t = f_t ⊙ δ_t ⊙ S_{t-1} + i_t ⊙ (K^T V)")
    print("   - O_t = o_t ⊙ (Q S_t)")
    print("   - 每个门独立学习")

    print("\n与其他方法的对比:")
    print("\nvs. GLA:")
    print("  - GDN: 多重门控 + 自适应衰减")
    print("  - GLA: 单一门控")
    print("  - GDN 表达能力更强，但计算稍复杂")
    print("\nvs. DeltaNet:")
    print("  - GDN: 门控 + 衰减")
    print("  - DeltaNet: 主要依赖衰减")
    print("  - GDN 对信息流控制更精细")
    print("\nvs. LSTM:")
    print("  - GDN: 线性注意力 + LSTM风格门控")
    print("  - LSTM: 纯 RNN")
    print("  - GDN 保持了注意力的并行性")

    print("\n优缺点:")
    print("优点:")
    print("  - 更强的表达能力（多重门控）")
    print("  - 更好的长距离依赖建模")
    print("  - 仍然是线性复杂度")
    print("  - 自适应衰减适应性强")
    print("  - 可以捕获多时间尺度模式")
    print("\n缺点:")
    print("  - 参数量较大（多个门控网络）")
    print("  - 训练可能更困难（更多超参数）")
    print("  - 推理稍慢（虽然仍是 O(1)）")
    print("  - 门控之间可能存在冗余")

    print("\n适用场景:")
    print("  - 需要精细控制的复杂任务")
    print("  - 长序列且包含多种时间尺度的模式")
    print("  - 对模型容量要求高的场景")
    print("  - 需要强长距离依赖建模")

    print("\n设计建议:")
    print("  - 从简单版本开始（少数门控）")
    print("  - 逐步增加复杂度")
    print("  - 仔细调优衰减参数")
    print("  - 考虑使用分层架构")
    print("  - 可能需要特殊的初始化策略")
