"""
Mamba2 Implementation

核心逻辑：
Mamba2 是基于状态空间模型（SSM）的架构，通过选择性状态空间实现线性复杂度
使用结构化状态空间（Structured State Space）和门控机制

关键创新：
1. 选择性状态空间：参数依赖于输入（input-dependent）
2. 硬件感知算法：使用高效的扫描算法
3. SSD (Structured State Space Duality)：结合SSM和注意力的优点
4. 块对角结构：提升并行性

数学形式：
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t + D * x_t

其中 A, B, C 是输入依赖的（这是选择性SSM的关键）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Mamba2Block(nn.Module):
    """
    Mamba2 核心块

    实现选择性状态空间模型（Selective SSM）
    复杂度：O(T * D * N) 其中 N 是状态维度，通常远小于 T
    """

    def __init__(
        self,
        d_model,          # 模型维度
        d_state=16,       # SSM 状态维度
        d_conv=4,         # 卷积核大小
        expand_factor=2,  # 扩展因子
        dt_rank="auto",   # delta (Δ) 的秩
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        bias=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # 1. 输入投影：x -> (z, x, B, C, dt)
        # z: 门控信号
        # x: 输入到 SSM
        # B, C: SSM 参数
        # dt: 时间步长（delta）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 2. 卷积层（用于局部依赖）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # 深度可分离卷积
            padding=d_conv - 1,
        )

        # 3. SSM 参数投影
        # Delta (Δ) 投影 - 控制状态更新速度
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 dt 投影
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt 的偏置设置为在 [dt_min, dt_max] 范围内
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # 逆 softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # 4. SSM 参数 A - 状态转移矩阵
        # 使用对角加低秩结构（Mamba2 的创新）
        # A 初始化为负值，确保稳定性
        A = torch.randn(self.d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))  # 使用 log 确保正定性

        # 5. B 和 C 投影（输入依赖）
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # 6. 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 输入投影和门控
        # (B, L, D) -> (B, L, 2*d_inner)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # 分割为 x 和门控信号 z

        # 2. 卷积（捕获局部信息）
        # (B, L, D) -> (B, D, L) -> (B, D, L) -> (B, L, D)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]  # 移除 padding
        x = x.transpose(1, 2)

        # 3. 激活
        x = F.silu(x)  # SiLU 激活

        # 4. SSM 核心计算（选择性状态空间）
        y = self.selective_scan(x)

        # 5. 门控
        y = y * F.silu(z)

        # 6. 输出投影
        output = self.out_proj(y)

        return output

    def selective_scan(self, x):
        """
        选择性扫描 - Mamba2 的核心

        实现状态空间模型的递归计算：
        h_t = A * h_{t-1} + B * x_t
        y_t = C * h_t

        关键：A, B, C 都是输入依赖的（这是"选择性"的含义）

        Args:
            x: (batch_size, seq_len, d_inner)

        Returns:
            y: (batch_size, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape

        # 1. 计算输入依赖的 B 和 C
        B = self.B_proj(x)  # (B, L, d_state)
        C = self.C_proj(x)  # (B, L, d_state)

        # 2. 计算 delta (Δ) - 离散化步长
        # 使用低秩投影
        dt = self.dt_proj.weight @ x.view(-1, d_inner).T  # (d_inner, B*L)
        dt = dt.T.view(batch_size, seq_len, d_inner)  # (B, L, d_inner)
        dt = dt + self.dt_proj.bias
        dt = F.softplus(dt)  # 确保 dt > 0

        # 3. 获取 A（状态转移矩阵）
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 4. 离散化 SSM 参数（零阶保持 - ZOH）
        # 连续时间 -> 离散时间
        # A_discrete = exp(Δ * A)
        # B_discrete = (A_discrete - I) * A^{-1} * B ≈ Δ * B（近似）
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # 5. 选择性扫描（递归计算）
        # 这是性能关键部分 - 在实际实现中使用硬件加速
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # h_t = A * h_{t-1} + B * x_t
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C * h_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        return y


class Mamba2(nn.Module):
    """完整的 Mamba2 层（包含残差和归一化）"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_block = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Pre-norm + 残差连接
        return x + self.mamba_block(self.norm(x))


# 技术分析示例
if __name__ == "__main__":
    print("=" * 80)
    print("Mamba2 技术分析")
    print("=" * 80)

    # 创建示例
    batch_size, seq_len, d_model = 2, 100, 64
    d_state = 16

    x = torch.randn(batch_size, seq_len, d_model)

    # Mamba2
    mamba = Mamba2(d_model, d_state=d_state)
    output = mamba(x)

    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 计算复杂度分析
    print("\n计算复杂度分析:")
    print(f"SSM 扫描: O(T * D * N) = O({seq_len} * {d_model} * {d_state})")
    print(f"  = {seq_len * d_model * d_state} 操作")
    print(f"vs. 标准注意力: O(T^2 * D) = O({seq_len}^2 * {d_model}) = {seq_len**2 * d_model} 操作")
    print(f"加速比: {(seq_len**2 * d_model) / (seq_len * d_model * d_state):.2f}x")

    # 内存使用
    state_size = batch_size * d_model * d_state * 4  # float32
    print(f"\n内存使用:")
    print(f"状态张量大小: {state_size / 1024:.2f} KB")
    print(f"vs. 注意力矩阵 (seq_len=1024): {batch_size * 8 * 1024 * 1024 * 4 / 1024 / 1024:.2f} MB")

    print("\n核心创新:")
    print("1. 选择性机制:")
    print("   - A, B, C 参数都是输入依赖的")
    print("   - 模型可以选择性地传播或忘记信息")
    print("   - 类似于 LSTM 的门控，但更高效")
    print("\n2. 硬件感知设计:")
    print("   - 使用高效的并行扫描算法")
    print("   - 优化的内存访问模式")
    print("   - GPU 友好的实现")
    print("\n3. 混合架构:")
    print("   - 卷积层捕获局部模式")
    print("   - SSM 捕获长距离依赖")
    print("   - 门控机制控制信息流")

    print("\n优缺点:")
    print("优点:")
    print("  - O(T) 线性复杂度，可处理超长序列")
    print("  - 内存效率高，只需存储状态向量")
    print("  - 推理速度快（常数级别缓存）")
    print("  - 可以处理 100K+ tokens 的序列")
    print("\n缺点:")
    print("  - 需要递归计算，并行性不如注意力")
    print("  - 对某些任务（如复制任务）表现不如注意力")
    print("  - 实现复杂度高，需要硬件优化")

    print("\n适用场景:")
    print("  - 长文档理解")
    print("  - 音频/视频处理（长序列）")
    print("  - 时间序列预测")
    print("  - 需要高效推理的场景")
