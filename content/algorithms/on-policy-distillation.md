# On-Policy Distillation 详解

> 本文档深入解析 On-Policy Distillation 算法的原理、实现细节及其在大语言模型训练中的应用。
>
> **参考资源**:
> - [Thinking Machines Lab Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
> - [tinker-cookbook GitHub 仓库](https://github.com/thinking-machines-lab/tinker-cookbook)

---

## 目录

1. [核心概念](#核心概念)
2. [算法原理](#算法原理)
3. [数学推导](#数学推导)
4. [实现细节](#实现细节)
5. [与传统方法对比](#与传统方法对比)
6. [实验结果](#实验结果)
7. [代码解析](#代码解析)
8. [最佳实践](#最佳实践)

---

## 核心概念

### 什么是 On-Policy Distillation?

**On-Policy Distillation** 是一种训练范式，其中：

- **学生模型** (Student Model) 根据**自身策略**生成轨迹（trajectory）
- **教师模型** (Teacher Model) 对这些轨迹提供**密集的、token级别**的监督信号
- **学生模型**通过**反向KL散度损失** (Reverse KL Divergence) 更新参数，使其概率分布与教师模型对齐

### 关键特性

| 特性 | 描述 |
|-----|------|
| **On-Policy** | 学生模型在自己生成的状态分布上学习 |
| **密集反馈** | 每个token都获得教师模型的监督信号 |
| **模式寻找** | 反向KL促使学生专注于教师的高概率行为 |
| **避免暴露偏差** | 学生在自己会遇到的上下文中学习 |

---

## 算法原理

### 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                     On-Policy Distillation                  │
└─────────────────────────────────────────────────────────────┘

第 t 步:

1. [采样] 学生模型 π_θ 根据提示 x 生成完整序列
   y ~ π_θ(·|x)

2. [评估] 教师模型 π_T 计算该序列的对数概率
   log π_T(y|x) = Σ log π_T(y_i | x, y_<i)

3. [损失计算] 计算反向KL散度
   L_KL = E_{y~π_θ} [ D_KL(π_T(·|x,y_<i) || π_θ(·|x,y_<i)) ]

4. [参数更新] 梯度下降优化学生模型
   θ ← θ - α ∇_θ L_KL
```

### 核心优势

#### 1. **避免复合误差** (Compounding Error)

**Off-Policy 问题**:
- 学生模型在教师模型生成的状态分布上训练
- 部署时在自己的状态分布上运行
- 分布不匹配导致错误累积

**On-Policy 解决方案**:
```
Off-Policy:  训练分布 P(s|π_T) ≠ 测试分布 P(s|π_θ) → 误差累积
On-Policy:   训练分布 P(s|π_θ) = 测试分布 P(s|π_θ) → 无分布漂移
```

#### 2. **计算效率**

- **9-30倍 FLOPs 减少**: 相比传统 off-policy SFT
- **7-10倍 梯度步减少**: 相比强化学习 (RL)
- **一行代码修改**: 在 KL-regularized RL 基础上仅需交换正则化模型

---

## 数学推导

### 反向 KL 散度

#### 定义

对于两个分布 P (教师) 和 Q (学生)，反向KL散度定义为:

```
D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
            = E_{x~P} [ log P(x) - log Q(x) ]
```

#### 梯度推导

学生模型的损失函数：

```
L(θ) = E_{y~π_θ(·|x)} [ D_KL(π_T(·|x,y_<i) || π_θ(·|x,y_<i)) ]
```

对于序列中的每个位置 i:

```
L_i(θ) = Σ_v π_T(v|x,y_<i) [ log π_T(v|x,y_<i) - log π_θ(v|x,y_<i) ]
```

梯度：

```
∇_θ L_i = -E_{v~π_T} [ ∇_θ log π_θ(v|x,y_<i) ]
```

这等价于在教师分布下进行**最大似然估计** (MLE)。

### 为什么使用反向 KL？

#### 反向 KL vs 正向 KL

| 类型 | 公式 | 特性 | 行为 |
|-----|------|-----|------|
| **正向 KL** | D_KL(Q \|\| P) | 零规避 (zero-avoiding) | 覆盖所有模式，导致分散 |
| **反向 KL** | D_KL(P \|\| Q) | 模式寻找 (mode-seeking) | 聚焦单一行为，精确逼近 |

#### 数学解释

当 P(x) > 0 但 Q(x) ≈ 0:

- **正向 KL**:
  ```
  D_KL(Q || P) = Q(x) log(Q(x)/P(x)) ≈ 0
  ```
  惩罚较小，学生会分散概率质量

- **反向 KL**:
  ```
  D_KL(P || Q) = P(x) log(P(x)/Q(x)) → ∞
  ```
  惩罚巨大，学生被迫在 P(x) > 0 处分配概率

#### 实际效果

```
教师分布 P: 两个峰值 (模式A, 模式B)

正向 KL → 学生分布: [====模式A====]  [====模式B====]
                    (覆盖两个模式，但都不精确)

反向 KL → 学生分布: [████模式A████]  [    ]
                    (专注单一模式，精确复制)
```

### 折扣未来 KL (Discounted Future KL)

#### 标准目标

```
L = E_{y~π_θ} [ Σ_{i=1}^T D_KL(π_T(·|x,y_<i) || π_θ(·|x,y_<i)) ]
```

#### 折扣版本

```
L_discounted = E_{y~π_θ} [ Σ_{i=1}^T γ^{T-i} D_KL(π_T(·|x,y_<i) || π_θ(·|x,y_<i)) ]
```

其中 γ ∈ [0, 1] 是折扣因子。

#### 直觉

- **γ = 1**: 所有步骤权重相同（标准设置）
- **γ < 1**: 更重视早期token的对齐
- **效果**: 优先学习关键的初始决策步骤

---

## 实现细节

### 训练算法伪代码

```python
def on_policy_distillation(
    student_model,
    teacher_model,
    dataset,
    num_steps,
    learning_rate,
    kl_penalty,
    discount_factor=1.0
):
    optimizer = Adam(student_model.parameters(), lr=learning_rate)

    for step in range(num_steps):
        # 1. 采样批次提示
        prompts = dataset.sample_batch()

        # 2. 学生模型生成完整响应 (on-policy)
        with torch.no_grad():
            student_outputs = student_model.generate(prompts)

        # 3. 计算教师和学生的对数概率
        teacher_logprobs = teacher_model.compute_logprobs(
            prompts, student_outputs
        )
        student_logprobs = student_model.compute_logprobs(
            prompts, student_outputs
        )

        # 4. 计算反向 KL 损失
        kl_loss = 0
        for i in range(len(student_outputs)):
            # 每个 token 位置的 KL
            kl_per_token = compute_reverse_kl(
                teacher_logprobs[i],
                student_logprobs[i]
            )

            # 应用折扣
            discount = discount_factor ** (len(student_outputs) - i - 1)
            kl_loss += discount * kl_per_token

        # 5. 加权并反向传播
        loss = kl_penalty * kl_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return student_model
```

### 关键配置参数

#### tinker-cookbook 实现

```python
@dataclass
class OnPolicyDistillationConfig:
    # 模型配置
    student_model: str = "Qwen/Qwen3-8B-Base"
    teacher_model: str = "Qwen/Qwen3-32B-Instruct"

    # 数据集
    dataset: str = "deepmath"  # 或 "tulu3"

    # LoRA 配置
    lora_rank: int = 128
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # 训练超参数
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_steps: int = 100

    # KL 参数
    kl_penalty: float = 0.1
    kl_discount_factor: float = 1.0

    # 环境设置
    use_rewards: bool = False  # On-policy 不使用任务奖励
```

### 数据集配置

#### PromptOnlyDatasetBuilder

```python
class PromptOnlyDatasetBuilder:
    """
    构建仅包含提示的数据集，不包含预定义的响应。
    学生模型将生成自己的响应。
    """

    def __init__(self, dataset_name: str, groups_per_batch: int):
        self.dataset_name = dataset_name
        self.groups_per_batch = groups_per_batch

    def build(self):
        if self.dataset_name == "deepmath":
            return load_deepmath_prompts()  # 103K 样本
        elif self.dataset_name == "openthoughts3":
            return load_openthoughts3_prompts()  # 1.2M 样本
        elif self.dataset_name == "tulu3":
            return load_tulu3_sft_mix()
```

### 环境配置：无奖励设置

```python
class RewardFreeEnvironment:
    """
    On-Policy Distillation 使用无奖励环境。
    唯一的监督信号来自 KL 散度最小化。
    """

    def get_reward(self, response, ground_truth):
        return 0.0  # 不提供正确性奖励

    def get_format_reward(self, response):
        return 0.0  # 不提供格式奖励
```

**设计理念**:
- 传统 RL 需要精心设计的奖励函数
- On-Policy Distillation 完全依赖教师模型的隐式知识
- 避免奖励塑形 (reward shaping) 的复杂性

---

## 与传统方法对比

### 1. Off-Policy Supervised Fine-Tuning (SFT)

| 维度 | Off-Policy SFT | On-Policy Distillation |
|-----|---------------|----------------------|
| **训练数据** | 教师生成的固定数据集 | 学生动态生成的轨迹 |
| **分布匹配** | ❌ 训练≠测试分布 | ✅ 训练=测试分布 |
| **复合误差** | ❌ 存在 | ✅ 无 |
| **计算成本** | 高 (需要大规模数据集) | 低 (9-30x 减少) |
| **AIME'24 得分** | ~55% @ 3000 步 | ~65% @ 100 步 |

#### 实验对比 (tinker-cookbook)

```bash
# Off-Policy SFT
python -m tinker_cookbook.recipes.distillation.off_policy_reasoning \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=openthoughts3 \
    learning_rate=1e-3 \
    lora_rank=128

# 结果: AIME'24 ~ 55%, 训练 3000 步

# On-Policy Distillation
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=deepmath \
    learning_rate=1e-4 \
    lora_rank=128

# 结果: AIME'24 ~ 65%, 训练 100 步
```

**性能提升**:
- **准确率**: +10% (55% → 65%)
- **训练效率**: 30x 更快 (3000 步 → 100 步)

### 2. 强化学习 (RL)

| 维度 | RL (PPO/REINFORCE) | On-Policy Distillation |
|-----|-------------------|----------------------|
| **监督信号** | 稀疏奖励 (episode 结束) | 密集反馈 (每个 token) |
| **奖励设计** | ✅ 需要 | ❌ 不需要 |
| **梯度步数** | 高 | 低 (7-10x 减少) |
| **实现复杂度** | 高 (value network, PPO clip) | 低 (一行代码修改) |
| **方差** | 高 | 低 (教师提供确定性目标) |

#### 统一视角：KL-Regularized RL

传统 RL 目标:

```
L_RL = E_{y~π_θ} [ R(y) - β D_KL(π_ref || π_θ) ]
```

On-Policy Distillation:

```
L_OPD = -E_{y~π_θ} [ β D_KL(π_T || π_θ) ]
```

**关键差异**:
- RL: 用参考模型 `π_ref` 正则化，最大化奖励 `R(y)`
- OPD: 用教师模型 `π_T` 正则化，**奖励为 0**

**代码修改** (在 RL 基础上):

```python
# RL Implementation
loss = -rewards.mean() + kl_penalty * kl_div(ref_model, student_model)

# On-Policy Distillation (仅需修改正则化项)
loss = kl_penalty * kl_div(teacher_model, student_model)  # 移除奖励项
```

### 3. 行为克隆 (Behavior Cloning)

| 维度 | BC | On-Policy Distillation |
|-----|------|----------------------|
| **数据来源** | 专家演示 (固定) | 学生生成 (动态) |
| **覆盖范围** | 仅专家访问的状态 | 学生可能访问的所有状态 |
| **泛化能力** | 弱 (分布外失败) | 强 (在自身分布上训练) |

---

## 实验结果

### 推理任务性能 (AIME'24 基准)

#### 配置

- **学生模型**: Qwen3-8B-Base
- **教师模型**: Qwen3-32B-Instruct
- **数据集**: DeepMath (103K 数学问题)
- **LoRA**: rank=128, α=16
- **学习率**: 1e-4

#### 结果

| 方法 | 训练步数 | AIME'24 得分 | FLOPs (相对) |
|-----|---------|-------------|-------------|
| Off-Policy SFT | 3000 | 55% | 30x |
| On-Policy Distillation | 100 | 65% | 1x |

**关键发现**:
1. **30倍计算效率**提升
2. **+10%绝对准确率**提升
3. **100步**即达到收敛

### 个性化任务性能 (IF-Eval)

#### 场景

将内部文档知识蒸馏到学生模型，同时保持通用能力。

#### 配置

- **数据集**: 内部文档 + Tulu3 (SFT 混合)
- **教师模型**: Qwen3-235B-Instruct
- **初始化**: 用 SFT 预热

#### 结果

| 指标 | 初始化 | 100 步后 |
|-----|--------|---------|
| IF-Eval | 基线 | 恢复到基线 |
| 内部任务准确率 | 0% | 85%+ |

**结论**: On-Policy Distillation 能快速适应新领域而不损害通用能力。

### 多教师蒸馏

#### 配置

```python
# 教师1: DeepMath 专家
teacher1 = TeacherConfig(
    model="Qwen/Qwen3-32B-Instruct",
    dataset="deepmath",
    groups_per_batch=2
)

# 教师2: 通用指令专家
teacher2 = TeacherConfig(
    model="Qwen/Qwen3-235B-Instruct",
    dataset="tulu3",
    groups_per_batch=2
)
```

#### 结果

| 单教师 (DeepMath) | 多教师 (DeepMath + Tulu3) |
|-----------------|--------------------------|
| AIME'24: 65% | AIME'24: 63% |
| IF-Eval: -15% | IF-Eval: -2% |

**权衡**: 轻微的推理性能下降换取显著的通用能力保持。

---

## 代码解析

### on_policy_distillation.py 核心组件

#### 1. 配置管理

```python
@dataclass
class CLIConfig:
    """命令行接口配置"""

    # 模型
    model_name: str = "Qwen/Qwen3-8B-Base"
    teacher_model_name: str = "Qwen/Qwen3-32B-Instruct"

    # 数据集: "deepmath" 或 "tulu3"
    dataset: str = "deepmath"

    # LoRA
    lora_rank: int = 128
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 训练
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_steps: int = 100
    gradient_accumulation_steps: int = 1

    # KL 参数
    kl_penalty: float = 0.1
    kl_discount_factor: float = 1.0

    # 日志
    log_dir: str = "./logs"
    save_every: int = 50
```

#### 2. 数据集构建

```python
def create_dataset_builder(config: CLIConfig) -> PromptOnlyDatasetBuilder:
    """创建数据集构建器"""

    if config.dataset == "deepmath":
        return PromptOnlyDatasetBuilder(
            dataset_name="deepmath",
            groups_per_batch=config.batch_size
        )
    elif config.dataset == "tulu3":
        return PromptOnlyDatasetBuilder(
            dataset_name="tulu3",
            groups_per_batch=config.batch_size
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
```

#### 3. 教师模型配置

```python
def create_teacher_config(config: CLIConfig) -> TeacherConfig:
    """配置教师模型"""

    return TeacherConfig(
        model_name=config.teacher_model_name,
        load_in_8bit=True,  # 节省内存
        torch_dtype=torch.bfloat16
    )
```

#### 4. 训练主循环

```python
async def train(config: CLIConfig):
    """主训练循环"""

    # 1. 初始化模型
    student = load_student_model(config)
    teacher = load_teacher_model(config)

    # 2. 创建数据加载器
    dataset = create_dataset_builder(config).build()

    # 3. 优化器
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate
    )

    # 4. 训练循环
    for step in range(config.num_steps):
        # 采样 prompts
        prompts = dataset.sample(config.batch_size)

        # 学生生成 (on-policy)
        with torch.no_grad():
            student_outputs = student.generate(
                prompts,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0
            )

        # 计算对数概率
        teacher_logprobs = teacher.compute_logprobs(
            prompts, student_outputs
        )
        student_logprobs = student.compute_logprobs(
            prompts, student_outputs
        )

        # KL 损失
        kl_loss = compute_reverse_kl_loss(
            teacher_logprobs,
            student_logprobs,
            discount_factor=config.kl_discount_factor
        )

        # 反向传播
        loss = config.kl_penalty * kl_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 日志
        if step % 10 == 0:
            print(f"Step {step}: KL Loss = {kl_loss.item():.4f}")

        # 保存检查点
        if step % config.save_every == 0:
            save_checkpoint(student, f"{config.log_dir}/step_{step}")
```

#### 5. KL 损失计算

```python
def compute_reverse_kl_loss(
    teacher_logprobs: torch.Tensor,  # [batch, seq_len, vocab]
    student_logprobs: torch.Tensor,  # [batch, seq_len, vocab]
    discount_factor: float = 1.0
) -> torch.Tensor:
    """
    计算反向 KL 散度: D_KL(teacher || student)

    Args:
        teacher_logprobs: 教师模型的对数概率
        student_logprobs: 学生模型的对数概率
        discount_factor: 折扣因子 γ

    Returns:
        加权 KL 散度损失
    """

    # 转换为概率分布
    teacher_probs = torch.softmax(teacher_logprobs, dim=-1)
    student_log_probs = torch.log_softmax(student_logprobs, dim=-1)

    # 计算 KL: Σ P(x) * log(P(x) / Q(x))
    #        = Σ P(x) * (log P(x) - log Q(x))
    kl_per_token = torch.sum(
        teacher_probs * (
            torch.log(teacher_probs + 1e-10) - student_log_probs
        ),
        dim=-1  # 在词汇表维度求和
    )  # [batch, seq_len]

    # 应用折扣
    seq_len = kl_per_token.shape[1]
    discounts = torch.tensor([
        discount_factor ** (seq_len - i - 1)
        for i in range(seq_len)
    ]).to(kl_per_token.device)

    discounted_kl = kl_per_token * discounts  # [batch, seq_len]

    # 平均
    return discounted_kl.mean()
```

### on_policy_multi_teacher.py 多教师实现

#### 配置结构

```python
@dataclass
class MultiTeacherConfig:
    """多教师蒸馏配置"""

    student_model: str = "Qwen/Qwen3-8B-Base"

    # 教师配置列表
    teachers: list[TeacherConfig] = field(default_factory=list)

    # 数据集配置列表
    datasets: list[DatasetConfig] = field(default_factory=list)

    # 其他参数...
```

#### 教师-数据集对

```python
# 示例: 双教师配置
config = MultiTeacherConfig(
    student_model="Qwen/Qwen3-8B-Base",
    teachers=[
        TeacherConfig(
            model_name="Qwen/Qwen3-32B-Instruct",
            load_in_8bit=True
        ),
        TeacherConfig(
            model_name="Qwen/Qwen3-235B-Instruct",
            load_in_8bit=True
        )
    ],
    datasets=[
        DatasetConfig(
            name="deepmath",
            groups_per_batch=2
        ),
        DatasetConfig(
            name="tulu3",
            groups_per_batch=2
        )
    ]
)
```

#### 训练循环修改

```python
async def train_multi_teacher(config: MultiTeacherConfig):
    """多教师训练"""

    # 1. 加载所有教师
    teachers = [
        load_teacher(teacher_config)
        for teacher_config in config.teachers
    ]

    # 2. 加载所有数据集
    datasets = [
        load_dataset(dataset_config)
        for dataset_config in config.datasets
    ]

    # 3. 训练循环
    for step in range(config.num_steps):
        total_loss = 0

        # 从每个教师-数据集对采样
        for teacher, dataset in zip(teachers, datasets):
            prompts = dataset.sample(dataset.groups_per_batch)

            # 学生生成
            student_outputs = student.generate(prompts)

            # 计算该教师的 KL 损失
            teacher_logprobs = teacher.compute_logprobs(
                prompts, student_outputs
            )
            student_logprobs = student.compute_logprobs(
                prompts, student_outputs
            )

            kl_loss = compute_reverse_kl_loss(
                teacher_logprobs, student_logprobs
            )

            total_loss += kl_loss

        # 联合优化
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 最佳实践

### 1. 超参数调优

#### 学习率

```python
# 推荐值
learning_rate = {
    "on_policy_distillation": 1e-4,  # 比 SFT 低 10x
    "off_policy_sft": 1e-3
}
```

**原因**: On-Policy 的梯度更稳定（教师提供确定性目标），可用更小学习率获得更精确对齐。

#### KL 惩罚系数

```python
# 典型范围
kl_penalty = [0.01, 0.1, 0.5, 1.0]

# 效果
# 小值 (0.01): 学生更自由，可能偏离教师
# 大值 (1.0):  学生严格模仿教师，可能过拟合
```

**调优策略**:
1. 从 0.1 开始
2. 监控学生生成质量
3. 如果输出质量下降，增大系数
4. 如果学生缺乏多样性，减小系数

#### 折扣因子

```python
# γ = 1.0: 无折扣（标准）
# γ = 0.99: 轻微偏向早期 token
# γ = 0.9:  强烈偏向早期 token

discount_factor = 1.0  # 推荐默认值
```

**使用场景**:
- **推理任务**: γ = 0.95 (早期步骤更关键)
- **生成任务**: γ = 1.0 (所有 token 同等重要)

### 2. 数据集选择

#### DeepMath

- **任务**: 数学推理
- **规模**: 103K 样本
- **适用**: 专注于单一领域的高性能

#### Tulu3

- **任务**: 通用指令跟随
- **规模**: SFT 混合数据集
- **适用**: 保持广泛能力

#### OpenThoughts3

- **任务**: 思维链推理
- **规模**: 1.2M 样本
- **适用**: Off-Policy SFT 预训练

**建议**:
```python
# 阶段1: Off-Policy SFT 预热
dataset = "openthoughts3"
method = "off_policy_sft"
steps = 3000

# 阶段2: On-Policy 精炼
dataset = "deepmath"
method = "on_policy_distillation"
steps = 100
```

### 3. LoRA 配置

```python
# 推荐配置
lora_config = {
    "rank": 128,           # 平衡表达力和效率
    "alpha": 16,           # α = rank / 8
    "dropout": 0.05,       # 轻微正则化
    "target_modules": [
        "q_proj",          # Query 投影
        "k_proj",          # Key 投影
        "v_proj",          # Value 投影
        "o_proj"           # Output 投影
    ]
}
```

**Rank 选择**:
- **32**: 快速实验 (内存受限)
- **128**: 生产环境 (推荐)
- **256**: 最大性能 (大规模模型)

### 4. 监控指标

#### 训练期间

```python
metrics_to_track = {
    "kl_divergence": "每步的平均 KL",
    "perplexity": "学生模型困惑度",
    "generation_diversity": "输出多样性 (distinct-n)",
    "teacher_student_agreement": "top-k token 一致率"
}
```

#### 评估期间

```python
eval_benchmarks = {
    "reasoning": "AIME, MATH, GSM8K",
    "instruction_following": "IF-Eval, MT-Bench",
    "general": "MMLU, HellaSwag"
}
```

### 5. 常见陷阱

#### ❌ 陷阱1: 教师模型过强

```python
# 问题: 学生无法逼近教师
student = "Qwen3-1.8B"
teacher = "Qwen3-235B"  # 容量差距过大

# 解决方案
student = "Qwen3-8B"
teacher = "Qwen3-32B"   # 合理的容量比
```

#### ❌ 陷阱2: 批次大小过小

```python
# 问题: 梯度估计方差大
batch_size = 1

# 解决方案
batch_size = 4
gradient_accumulation_steps = 4  # 有效批次 = 16
```

#### ❌ 陷阱3: 过度训练

```python
# 问题: 学生过拟合教师的分布偏差
num_steps = 1000  # 过多

# 解决方案
num_steps = 100
# 监控验证集性能，及早停止
```

### 6. 生产部署

#### 检查点管理

```python
# 保存策略
save_every = 50
save_total_limit = 3  # 仅保留最近 3 个检查点

# 验证检查点
def validate_checkpoint(checkpoint_path, eval_dataset):
    model = load_checkpoint(checkpoint_path)
    metrics = evaluate(model, eval_dataset)
    return metrics["accuracy"]

# 选择最佳检查点
best_checkpoint = max(
    checkpoints,
    key=lambda ckpt: validate_checkpoint(ckpt, eval_data)
)
```

#### 合并 LoRA 权重

```python
from peft import PeftModel

# 加载基础模型和 LoRA
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")
lora_model = PeftModel.from_pretrained(base_model, "path/to/lora")

# 合并
merged_model = lora_model.merge_and_unload()

# 保存
merged_model.save_pretrained("path/to/merged_model")
```

---

## 高级话题

### 1. 理论基础

#### 分布偏移分析

定义学生策略在第 t 步的状态分布为 ρ^π_θ_t，教师策略的状态分布为 ρ^π_T。

**Off-Policy 分布差异**:

```
d_TV(ρ^π_θ_T, ρ^π_T) ≤ Σ_{t=0}^T γ^t E_{s~ρ^π_θ_t} [ d_TV(π_θ(·|s), π_T(·|s)) ]
```

随着时间步 T 增加，分布偏移以 γ^T 指数增长（复合误差）。

**On-Policy 解决**:

```
在每步更新后, ρ^π_θ ≈ ρ^π_T
→ d_TV(ρ^π_θ_T, ρ^π_T) ≈ 0
```

#### 收敛保证

在以下假设下，On-Policy Distillation 收敛到教师策略：

1. **充分探索**: 学生策略 π_θ 访问所有教师访问的状态
2. **Lipschitz 连续**: KL 散度关于 θ 连续
3. **学习率衰减**: Σ α_t = ∞, Σ α_t^2 < ∞

**定理** (非正式):

```
lim_{t→∞} E_{s~ρ^π_θ} [ D_KL(π_T(·|s) || π_θ(·|s)) ] = 0
```

### 2. 变体与扩展

#### 2.1 Top-K KL 蒸馏

仅对教师分布的 top-k 高概率 token 计算 KL：

```python
def top_k_kl_loss(teacher_logits, student_logits, k=50):
    """仅蒸馏教师的 top-k token"""

    # 获取教师的 top-k
    top_k_indices = torch.topk(teacher_logits, k, dim=-1).indices

    # 提取对应的 logits
    teacher_top_k = torch.gather(teacher_logits, -1, top_k_indices)
    student_top_k = torch.gather(student_logits, -1, top_k_indices)

    # 计算 KL
    teacher_probs = torch.softmax(teacher_top_k, dim=-1)
    student_log_probs = torch.log_softmax(student_top_k, dim=-1)

    kl = torch.sum(
        teacher_probs * (torch.log(teacher_probs) - student_log_probs),
        dim=-1
    )

    return kl.mean()
```

**优势**: 减少计算量，聚焦于关键 token。

#### 2.2 温度缩放

调整教师分布的"锐度"：

```python
def temperature_scaled_kl(teacher_logits, student_logits, temperature=2.0):
    """使用温度缩放的 KL 蒸馏"""

    # 教师分布平滑
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)

    kl = torch.sum(
        teacher_probs * (torch.log(teacher_probs) - student_log_probs),
        dim=-1
    )

    return kl.mean()
```

**效果**:
- T > 1: 平滑分布，传递更多"暗知识"
- T = 1: 标准 KL
- T < 1: 锐化分布，更保守

#### 2.3 混合目标

结合 on-policy 蒸馏和任务奖励：

```python
def mixed_objective(
    teacher_logprobs,
    student_logprobs,
    rewards,
    kl_weight=0.8,
    reward_weight=0.2
):
    """混合 KL 蒸馏和奖励优化"""

    # KL 项
    kl_loss = compute_reverse_kl_loss(teacher_logprobs, student_logprobs)

    # 奖励项 (REINFORCE)
    reward_loss = -torch.mean(student_logprobs * rewards)

    # 加权组合
    total_loss = kl_weight * kl_loss + reward_weight * reward_loss

    return total_loss
```

**使用场景**: 当教师模型不完美时，任务奖励可提供额外监督。

### 3. 扩展到多模态

#### 视觉-语言模型蒸馏

```python
def multimodal_on_policy_distillation(
    student_vlm,
    teacher_vlm,
    image_text_pairs
):
    """多模态 On-Policy 蒸馏"""

    for images, prompts in image_text_pairs:
        # 学生生成
        student_outputs = student_vlm.generate(
            images=images,
            prompts=prompts
        )

        # 教师评分
        teacher_logprobs = teacher_vlm.compute_logprobs(
            images=images,
            prompts=prompts,
            outputs=student_outputs
        )

        student_logprobs = student_vlm.compute_logprobs(
            images=images,
            prompts=prompts,
            outputs=student_outputs
        )

        # KL 损失
        loss = compute_reverse_kl_loss(teacher_logprobs, student_logprobs)
        loss.backward()
```

**挑战**:
- 图像编码器的对齐
- 视觉-文本注意力的蒸馏

---

## 总结

### 核心要点

1. **On-Policy 的本质**: 学生在自己的分布上学习，避免分布偏移
2. **反向 KL 的作用**: 模式寻找，精确复制教师行为
3. **计算效率**: 9-30x FLOPs 减少，7-10x 梯度步减少
4. **无需奖励**: 完全依赖教师隐式监督
5. **易于实现**: 在 RL 代码基础上一行修改

### 适用场景

✅ **推荐使用**:
- 有强教师模型但计算受限
- 需要快速适应新领域
- 避免奖励工程
- 推理/生成任务

❌ **不推荐**:
- 教师模型本身性能不佳
- 需要超越教师（探索新策略）
- 离线数据集已充分覆盖目标分布

### 未来方向

1. **自适应 KL 权重**: 根据训练阶段动态调整
2. **多任务蒸馏**: 单个学生模型蒸馏多个专家教师
3. **层级蒸馏**: 同时对齐中间层表示
4. **在线教师改进**: 教师模型随学生进步而更新

---

## 参考资源

### 论文

1. **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**
   arXiv:2306.13649

2. **Distilling Policy Distillation**
   AISTATS 2019

### 代码库

- [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Hugging Face PEFT](https://github.com/huggingface/peft)

### 博客

- [Thinking Machines Lab: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)

---

## 附录

### A. 完整训练脚本示例

```bash
#!/bin/bash

# 配置
STUDENT_MODEL="Qwen/Qwen3-8B-Base"
TEACHER_MODEL="Qwen/Qwen3-32B-Instruct"
DATASET="deepmath"
OUTPUT_DIR="./outputs/on_policy_distillation"

# 运行 On-Policy Distillation
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=${STUDENT_MODEL} \
    teacher_model_name=${TEACHER_MODEL} \
    dataset=${DATASET} \
    learning_rate=1e-4 \
    lora_rank=128 \
    lora_alpha=16 \
    kl_penalty=0.1 \
    kl_discount_factor=1.0 \
    batch_size=4 \
    num_steps=100 \
    log_dir=${OUTPUT_DIR} \
    save_every=50

# 评估
python -m tinker_cookbook.eval.aime \
    --model_path ${OUTPUT_DIR}/final_model \
    --output_file ${OUTPUT_DIR}/aime_results.json
```

### B. 数学符号表

| 符号 | 含义 |
|-----|------|
| π_θ | 学生策略 (参数化为 θ) |
| π_T | 教师策略 |
| π_ref | 参考策略 (RL 中使用) |
| D_KL(P \|\| Q) | P 到 Q 的 KL 散度 |
| ρ^π | 策略 π 的状态分布 |
| γ | 折扣因子 |
| α | 学习率 |
| β | KL 惩罚系数 |
| R(y) | 响应 y 的奖励 |
| T | 序列长度 |

### C. 常见问题 (FAQ)

**Q1: On-Policy Distillation 能否超越教师模型？**

A: 一般不能。它的目标是近似教师，而非探索新策略。若需超越，考虑：
- 混合 RL 奖励（混合目标）
- 使用多个互补教师
- 迭代式蒸馏（学生成为新教师）

**Q2: 如何选择教师模型大小？**

A: 经验法则：
```
教师参数量 ≈ 2x ~ 5x 学生参数量
```

例如：
- 学生 8B → 教师 32B ✅
- 学生 8B → 教师 235B ⚠️ (可能过强)
- 学生 8B → 教师 14B ✅

**Q3: On-Policy 是否总是优于 Off-Policy？**

A: 不一定。权衡：

| 方法 | 优势 | 劣势 |
|-----|------|------|
| On-Policy | 无分布偏移，高效 | 需要在线生成 |
| Off-Policy | 可重用数据集，稳定 | 复合误差 |

**推荐**: Off-Policy SFT 预热 → On-Policy 精炼

**Q4: 如何调试 KL 损失不下降？**

检查清单：
1. 学习率是否过大/过小？
2. 教师-学生容量差距是否合理？
3. 学生生成是否过于随机（temperature 过高）？
4. 批次大小是否足够？

---

**文档版本**: v1.0
**最后更新**: 2025-11-11
**维护者**: Thinking Machines Lab
