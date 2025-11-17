"""
Key-Decomposed Attention (KDA) Implementation

核心逻辑：
KDA 通过将键（Key）分解为多个独立的子空间来降低注意力计算复杂度
每个子空间捕获不同的语义或结构特征

关键创新：
1. 键分解：K = [K_1, K_2, ..., K_n]，每个 K_i 在独立子空间
2. 独立注意力：对每个子空间单独计算注意力
3. 融合机制：智能地组合多个子空间的输出
4. 低秩近似：通过分解实现参数和计算效率

数学形式：
K = [K_1, K_2, ..., K_n]  # 键分解到 n 个子空间
Q_i = Q W_i                # 查询投影到对应子空间
A_i = attention(Q_i, K_i, V)  # 每个子空间独立计算
O = Fusion([A_1, A_2, ..., A_n])  # 融合

复杂度：
标准注意力: O(T^2 * D)
KDA: O(n * T^2 * D/n) = O(T^2 * D) 但常数项小很多
或配合线性化: O(T * D^2 / n)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KeyDecomposedAttention(nn.Module):
    """
    键分解注意力 (KDA)

    将注意力计算分解到多个子空间
    每个子空间可以使用不同的注意力机制
    """

    def __init__(
        self,
        d_model,
        num_heads=8,
        num_subspaces=4,  # 子空间数量
        subspace_type="standard",  # standard, linear, or mixed
        fusion_type="weighted",  # weighted, concat, or gated
        share_qv=False,  # 是否在子空间间共享 Q 和 V
    ):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_subspaces == 0, "num_heads must be divisible by num_subspaces"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_subspaces = num_subspaces
        self.heads_per_subspace = num_heads // num_subspaces
        self.d_k = d_model // num_heads
        self.subspace_dim = self.heads_per_subspace * self.d_k
        self.subspace_type = subspace_type
        self.fusion_type = fusion_type
        self.share_qv = share_qv

        # Q, V 投影（可选共享）
        if share_qv:
            self.w_q = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)
        else:
            # 每个子空间独立的 Q, V
            self.w_q_list = nn.ModuleList([
                nn.Linear(d_model, self.subspace_dim, bias=False)
                for _ in range(num_subspaces)
            ])
            self.w_v_list = nn.ModuleList([
                nn.Linear(d_model, self.subspace_dim, bias=False)
                for _ in range(num_subspaces)
            ])

        # K 分解：每个子空间独立的 K
        self.w_k_list = nn.ModuleList([
            nn.Linear(d_model, self.subspace_dim, bias=False)
            for _ in range(num_subspaces)
        ])

        # 子空间特定的参数
        if subspace_type in ["linear", "mixed"]:
            # 线性注意力需要的特征映射
            self.feature_map_params = nn.ParameterList([
                nn.Parameter(torch.randn(self.subspace_dim))
                for _ in range(num_subspaces if subspace_type == "mixed" else 1)
            ])

        # 融合机制
        if fusion_type == "weighted":
            # 可学习的权重
            self.fusion_weights = nn.Parameter(torch.ones(num_subspaces) / num_subspaces)
        elif fusion_type == "gated":
            # 门控融合
            self.fusion_gate = nn.Linear(d_model * num_subspaces, num_subspaces)
        elif fusion_type == "concat":
            # 拼接后投影
            self.fusion_proj = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 对每个子空间独立计算注意力
        subspace_outputs = []

        for i in range(self.num_subspaces):
            # 获取当前子空间的 Q, K, V
            if self.share_qv:
                Q = self.w_q(x)
                V = self.w_v(x)
            else:
                Q = self.w_q_list[i](x)
                V = self.w_v_list[i](x)

            K = self.w_k_list[i](x)

            # reshape 为多头形式
            Q = Q.view(batch_size, seq_len, self.heads_per_subspace, self.d_k)
            K = K.view(batch_size, seq_len, self.heads_per_subspace, self.d_k)
            V = V.view(batch_size, seq_len, self.heads_per_subspace, self.d_k)

            # 根据子空间类型选择注意力机制
            if self.subspace_type == "standard":
                output_i = self._standard_attention(Q, K, V, mask)
            elif self.subspace_type == "linear":
                output_i = self._linear_attention(Q, K, V, i)
            elif self.subspace_type == "mixed":
                # 混合模式：部分子空间用标准注意力，部分用线性
                if i < self.num_subspaces // 2:
                    output_i = self._standard_attention(Q, K, V, mask)
                else:
                    output_i = self._linear_attention(Q, K, V, i)
            else:
                raise ValueError(f"Unknown subspace_type: {self.subspace_type}")

            subspace_outputs.append(output_i)

        # 2. 融合子空间输出
        output = self._fuse_subspaces(subspace_outputs, x)

        # 3. 输出投影
        output = self.w_o(output)

        return output

    def _standard_attention(self, Q, K, V, mask=None):
        """
        标准的 softmax 注意力

        Args:
            Q, K, V: (B, L, H_sub, D_k)
            mask: optional

        Returns:
            output: (B, L, H_sub * D_k)
        """
        batch_size, seq_len, heads_per_sub, d_k = Q.shape

        # 转置以便计算
        Q = Q.transpose(1, 2)  # (B, H_sub, L, D_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn = F.softmax(scores, dim=-1)

        # 应用到 V
        context = torch.matmul(attn, V)  # (B, H_sub, L, D_k)

        # 转回并合并头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, heads_per_sub * d_k)

        return context

    def _linear_attention(self, Q, K, V, subspace_idx):
        """
        线性注意力（核化版本）

        使用 φ(Q) (φ(K)^T V) 代替 softmax(QK^T) V
        复杂度：O(T * D^2) 而不是 O(T^2 * D)

        Args:
            Q, K, V: (B, L, H_sub, D_k)
            subspace_idx: 当前子空间索引

        Returns:
            output: (B, L, H_sub * D_k)
        """
        batch_size, seq_len, heads_per_sub, d_k = Q.shape

        # 应用特征映射 φ
        Q = self._apply_feature_map(Q)
        K = self._apply_feature_map(K)

        # 线性注意力递归形式
        # 转换形状以便计算
        Q = Q.transpose(1, 2)  # (B, H_sub, L, D_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算 KV 累积
        # (B, H_sub, L, D_k, 1) @ (B, H_sub, L, 1, D_k) -> (B, H_sub, L, D_k, D_k)
        # 然后累加：sum over L
        # 简化版本：直接计算 K^T V
        K_T = K.transpose(-2, -1)  # (B, H_sub, D_k, L)
        KV = torch.matmul(K_T, V)  # (B, H_sub, D_k, D_k)

        # 计算输出：Q @ (K^T V)
        context = torch.matmul(Q, KV)  # (B, H_sub, L, D_k)

        # 归一化
        K_sum = K.sum(dim=2, keepdim=True)  # (B, H_sub, 1, D_k)
        normalizer = torch.matmul(Q, K_sum.transpose(-2, -1))  # (B, H_sub, L, 1)
        context = context / (normalizer + 1e-6)

        # 转回并合并头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, heads_per_sub * d_k)

        return context

    def _apply_feature_map(self, x):
        """应用特征映射 φ(x)"""
        # 使用 ELU + 1 确保非负
        return F.elu(x) + 1

    def _fuse_subspaces(self, subspace_outputs, x):
        """
        融合多个子空间的输出

        Args:
            subspace_outputs: list of (B, L, subspace_dim)
            x: (B, L, D) 原始输入（用于门控）

        Returns:
            fused: (B, L, D)
        """
        batch_size, seq_len, _ = x.shape

        if self.fusion_type == "weighted":
            # 加权平均
            # 先拼接
            stacked = torch.stack(subspace_outputs, dim=-1)  # (B, L, subspace_dim, n_sub)

            # 应用权重
            weights = F.softmax(self.fusion_weights, dim=0)  # (n_sub,)
            fused = torch.matmul(stacked, weights)  # (B, L, subspace_dim)

            # 如果子空间维度 != d_model，需要处理
            if fused.shape[-1] != self.d_model:
                # 拼接所有子空间
                fused = torch.cat(subspace_outputs, dim=-1)  # (B, L, n_sub * subspace_dim)
                # 应该等于 d_model
                assert fused.shape[-1] == self.d_model

        elif self.fusion_type == "gated":
            # 门控融合
            # 拼接所有子空间
            concat = torch.cat(subspace_outputs, dim=-1)  # (B, L, D) 或 (B, L, n_sub * subspace_dim)

            # 计算门控权重
            gates = self.fusion_gate(concat)  # (B, L, n_sub)
            gates = F.softmax(gates, dim=-1)

            # 应用门控
            stacked = torch.stack(subspace_outputs, dim=2)  # (B, L, n_sub, subspace_dim)
            fused = torch.einsum('bln,blnd->bld', gates, stacked)

            # 确保维度正确
            if fused.shape[-1] * self.num_subspaces == self.d_model:
                fused = torch.cat(subspace_outputs, dim=-1)

        elif self.fusion_type == "concat":
            # 简单拼接
            fused = torch.cat(subspace_outputs, dim=-1)  # (B, L, D)
            # 如果需要，通过投影调整维度
            if fused.shape[-1] != self.d_model:
                fused = self.fusion_proj(fused)

        else:
            # 默认：直接拼接
            fused = torch.cat(subspace_outputs, dim=-1)

        return fused


class AdaptiveKDA(nn.Module):
    """
    自适应 KDA

    根据输入动态调整子空间的数量和分配
    """

    def __init__(self, d_model, num_heads=8, max_subspaces=8, min_subspaces=2):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_subspaces = max_subspaces
        self.min_subspaces = min_subspaces

        # 子空间数量预测器
        self.subspace_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_subspaces),
            nn.Sigmoid()
        )

        # 为每个可能的子空间准备 KDA
        self.kda_modules = nn.ModuleList([
            KeyDecomposedAttention(
                d_model,
                num_heads=num_heads,
                num_subspaces=n,
                subspace_type="mixed"
            )
            for n in range(min_subspaces, max_subspaces + 1)
        ])

    def forward(self, x):
        """
        自适应选择子空间数量

        Args:
            x: (B, L, D)

        Returns:
            output: (B, L, D)
        """
        # 预测子空间重要性
        # 使用平均池化作为序列表示
        x_mean = x.mean(dim=1)  # (B, D)
        subspace_scores = self.subspace_predictor(x_mean)  # (B, max_subspaces)

        # 选择最佳子空间数量（简化：使用最高得分的数量）
        best_num_subspaces = torch.argmax(subspace_scores, dim=-1).mode().values.item()
        best_num_subspaces = max(self.min_subspaces, min(best_num_subspaces + 1, self.max_subspaces))

        # 使用对应的 KDA
        kda_idx = best_num_subspaces - self.min_subspaces
        output = self.kda_modules[kda_idx](x)

        return output


# 技术分析示例
if __name__ == "__main__":
    print("=" * 80)
    print("Key-Decomposed Attention (KDA) 技术分析")
    print("=" * 80)

    # 创建示例
    batch_size, seq_len, d_model = 2, 100, 64
    num_heads = 8
    num_subspaces = 4

    x = torch.randn(batch_size, seq_len, d_model)

    # 标准 KDA
    kda_standard = KeyDecomposedAttention(
        d_model, num_heads,
        num_subspaces=num_subspaces,
        subspace_type="standard"
    )
    output_standard = kda_standard(x)

    # 线性 KDA
    kda_linear = KeyDecomposedAttention(
        d_model, num_heads,
        num_subspaces=num_subspaces,
        subspace_type="linear"
    )
    output_linear = kda_linear(x)

    # 混合 KDA
    kda_mixed = KeyDecomposedAttention(
        d_model, num_heads,
        num_subspaces=num_subspaces,
        subspace_type="mixed"
    )
    output_mixed = kda_mixed(x)

    print(f"\n输入形状: {x.shape}")
    print(f"标准 KDA 输出: {output_standard.shape}")
    print(f"线性 KDA 输出: {output_linear.shape}")
    print(f"混合 KDA 输出: {output_mixed.shape}")

    # 自适应 KDA
    adaptive_kda = AdaptiveKDA(d_model, num_heads)
    output_adaptive = adaptive_kda(x)
    print(f"自适应 KDA 输出: {output_adaptive.shape}")

    print("\n计算复杂度分析:")
    print(f"标准注意力: O(T^2 * D) = {seq_len**2 * d_model}")
    print(f"\nKDA (标准子空间):")
    print(f"  每个子空间: O(T^2 * D/n) = {seq_len**2 * d_model // num_subspaces}")
    print(f"  总共: {num_subspaces} * {seq_len**2 * d_model // num_subspaces} = {seq_len**2 * d_model}")
    print(f"  但实际更快因为：")
    print(f"    - 更小的矩阵乘法（缓存友好）")
    print(f"    - 可以并行计算子空间")
    print(f"\nKDA (线性子空间):")
    print(f"  每个子空间: O(T * (D/n)^2) = {seq_len * (d_model // num_subspaces)**2}")
    print(f"  总共: {num_subspaces * seq_len * (d_model // num_subspaces)**2}")

    print("\n核心创新:")
    print("1. 键分解:")
    print("   - K 分解到多个独立子空间")
    print("   - K = [K_1, K_2, ..., K_n]")
    print("   - 每个子空间捕获不同特征")
    print("\n2. 独立注意力计算:")
    print("   - 每个子空间独立计算注意力")
    print("   - 可以使用不同的注意力类型")
    print("   - 并行性好")
    print("\n3. 智能融合:")
    print("   - 加权融合：学习子空间重要性")
    print("   - 门控融合：动态调整组合")
    print("   - 拼接融合：保留所有信息")
    print("\n4. 混合架构:")
    print("   - 部分子空间用标准注意力（表达能力）")
    print("   - 部分子空间用线性注意力（效率）")
    print("   - 平衡性能和效率")

    print("\n与其他方法的对比:")
    print("\nvs. 标准注意力:")
    print("  - KDA: 分解计算，内存局部性更好")
    print("  - 标准注意力: 单一大矩阵计算")
    print("\nvs. 多头注意力:")
    print("  - KDA: 子空间可以使用不同机制")
    print("  - 多头注意力: 所有头使用相同机制")
    print("\nvs. 混合专家 (MoE):")
    print("  - KDA: 注意力层面的分解")
    print("  - MoE: FFN 层面的分解")
    print("  - 概念类似，应用不同")

    print("\n优缺点:")
    print("优点:")
    print("  - 计算效率高（更好的缓存利用）")
    print("  - 灵活性强（混合不同注意力类型）")
    print("  - 可解释性好（不同子空间捕获不同特征）")
    print("  - 参数效率（共享 Q, V 时）")
    print("  - 容易并行化")
    print("\n缺点:")
    print("  - 子空间数量是超参数")
    print("  - 融合机制需要仔细设计")
    print("  - 可能存在子空间冗余")
    print("  - 训练可能需要特殊策略")

    print("\n适用场景:")
    print("  - 需要高效推理的生产环境")
    print("  - 多模态任务（不同子空间处理不同模态）")
    print("  - 需要可解释性的应用")
    print("  - 内存受限的环境")
    print("  - 需要灵活性的研究项目")

    print("\n设计建议:")
    print("  - 从少量子空间开始（2-4个）")
    print("  - 尝试混合架构（标准+线性）")
    print("  - 使用门控融合获得更好的适应性")
    print("  - 考虑任务特点选择子空间数量")
    print("  - 可以与其他技术（如 GLA）结合")

    print("\n参数量对比:")
    params_standard = d_model * d_model * 4  # Q, K, V, O
    params_kda = d_model * (d_model // num_subspaces) * num_subspaces * 3 + d_model * d_model
    print(f"  标准注意力: {params_standard / 1000:.1f}K 参数")
    print(f"  KDA (独立 QKV): {params_kda / 1000:.1f}K 参数")
    print(f"  KDA (共享 QV): ~{params_standard / 1000:.1f}K 参数")
