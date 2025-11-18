# verl 技术解读文档

## 目录
- [项目概述](#项目概述)
- [一、训练 Tricks](#一训练-tricks)
- [二、数据准备](#二数据准备)
- [三、算法关键迭代](#三算法关键迭代)
- [四、架构设计](#四架构设计)
- [五、性能优化](#五性能优化)

---

## 项目概述

**verl (Volcano Engine Reinforcement Learning)** 是由字节跳动 Seed 团队开源的大语言模型强化学习训练库，是 HybridFlow 框架的生产级实现（论文已被 EuroSys 2025 接收）。

### 核心特性

- **灵活性**：支持 PPO、GRPO、GSPO、ReMax、RLOO、DAPO 等多种 RL 算法
- **高性能**：通过 3D-HybridEngine 实现最先进的训练吞吐量，相比 SOTA 基线提升 1.53× 性能
- **易集成**：无缝对接 FSDP、Megatron-LM、vLLM、SGLang 等主流基础设施
- **可扩展**：支持高达 671B 参数的模型训练，支持专家并行

### 显著成果

- **DAPO 系统**：基于 Qwen2.5-32B 在 AIME 2024 达到 50 分
- **Doubao-1.5-pro**：在数学基准测试中达到 OpenAI O1 级别性能（AIME 70.0 pass@1）
- **DeepSeek-R1 复现**：社区使用 verl 成功复现推理模型

---

## 一、训练 Tricks

### 1.1 DAPO 核心创新（2025 SOTA）

DAPO (Decouple Clip and Dynamic sAmpling Policy Optimization) 针对长思维链（long-CoT）场景提出四大关键技术：

#### 1.1.1 Clip-Higher 策略

**问题**：朴素 PPO/GRPO 训练中，策略熵快速下降导致熵坍塌（entropy collapse）

**解决方案**：
```yaml
# 配置示例
algorithm:
  clip_ratio_low: 0.2    # 标准下限裁剪
  clip_ratio_high: 0.28  # 更高的上限裁剪（关键创新）
```

**原理**：
- 传统 PPO 使用对称裁剪 `[1-ε, 1+ε]`
- Clip-Higher 使用非对称裁剪 `[1-0.2, 1+0.28]`，允许策略更大胆地探索高奖励区域
- 有效促进多样性，避免过早收敛

#### 1.1.2 Dynamic Sampling（动态采样）

**核心思想**：根据训练阶段动态调整每个 prompt 的采样数量

**实现策略**：
- 早期训练：增加采样数（如 n=16），充分探索
- 后期训练：减少采样数（如 n=8），降低 off-policy 偏差
- 提升训练效率和稳定性

#### 1.1.3 Token-Level Policy Gradient Loss

**配置**：
```yaml
actor_rollout_ref:
  actor:
    ppo_loss_type: "token-mean"  # DAPO/GRPO 推荐
    # 不使用 "seq-mean-token-mean"（原始 GRPO，长 CoT 场景不稳定）
```

**关键点**：
- **token-mean**：对所有 token 的损失求平均，适合长 CoT
- **seq-mean-token-mean**：先对每个序列平均再对 batch 平均，可能在长序列中不稳定
- **seq-mean-token-sum-norm** (DrGRPO)：使用全局常数归一化，消除长度偏差

#### 1.1.4 Overlong Reward Shaping

**目标**：减少奖励噪声，特别是超长序列

**技术**：
- 对超长生成序列进行奖励塑形
- 缓解长度偏差
- 提高奖励信号质量

### 1.2 多 PPO Epochs 技巧

**问题**：GRPO 存在 rank bias（排名偏差）

**解决方案**：
```yaml
actor_rollout_ref:
  actor:
    ppo_epochs: 4  # 增加优化步数（默认为 1）
```

**机制**：
- 第 1 步：高排名样本被推到裁剪阈值之外
- 后续步：强制关注低排名样本
- 有效缓解排名偏差

### 1.3 KL 散度控制

#### KL 散度类型

```yaml
actor_rollout_ref:
  actor:
    kl_loss_type: "kl"  # 可选: kl, abs, mse, low_var_kl, full
    kl_loss_coef: 0.001  # 推荐起始值
    use_kl_loss: True    # PPO: False, GRPO: True, DAPO: False
```

**不同类型说明**：
- **kl (k1)**：标准 KL 散度
- **abs**：绝对值损失
- **mse (k2)**：均方误差
- **low_var_kl (k3)**：低方差 KL
- **full**：完整 KL 散度
- **后缀 "+"**：应用 straight-through 使用 k2 进行无偏梯度估计

#### 关键配置差异

```yaml
# PPO 配置
algorithm:
  use_kl_in_reward: True
actor_rollout_ref.actor:
  use_kl_loss: False

# GRPO 配置
algorithm:
  use_kl_in_reward: False
actor_rollout_ref.actor:
  use_kl_loss: True

# DAPO 配置
algorithm:
  use_kl_in_reward: False
actor_rollout_ref.actor:
  use_kl_loss: False
```

### 1.4 推荐超参数（DAPO 最佳实践）

```yaml
# 优化器配置
optimizer:
  type: AdamW
  lr: 1e-6           # 常数学习率
  warmup_steps: 20   # 线性预热

# 批次配置
data:
  train_batch_size: 512              # 每次迭代的 prompt 数

actor_rollout:
  rollout:
    n: 16                            # 每个 prompt 采样 16 个回复

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 512         # mini-batch 大小
    ppo_micro_batch_size_per_gpu: 4  # 每个 GPU 的 micro-batch
    gradient_accumulation_steps: 16  # 每个 rollout 16 次梯度更新
```

### 1.5 其他重要 Tricks

#### 序列打包（Sequence Packing）

```yaml
# 适用于 Llama、Mistral、Gemma、Qwen 系列模型
actor_rollout_ref:
  actor:
    use_remove_padding: True
```

**效果**：消除 padding 开销，提升训练效率

#### Flash Attention 2

- 自动支持，显著降低内存占用
- 提升 attention 计算速度

#### LoRA 支持

```yaml
actor_rollout_ref:
  actor:
    lora:
      enabled: True
      r: 8
      alpha: 16
```

---

## 二、数据准备

### 2.1 数据格式要求

#### 存储格式

verl 要求数据以 **Parquet 格式**存储：

```python
# 保存数据集为 Parquet 格式
train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
```

#### 必需字段

1. **prompt 字段**：
   - 必须使用 HuggingFace `chat_template` 格式构建
   - RLHFDataset 会自动应用 chat template 并进行 tokenization

2. **可选字段**：
   - 奖励相关字段
   - 元数据字段

### 2.2 数据配置

```yaml
data:
  # 数据文件路径
  train_files:
    - /path/to/train.parquet

  # 字段配置
  prompt_key: "prompt"           # prompt 字段名称

  # 长度限制
  max_prompt_length: 2048        # 最大 prompt 长度
  max_response_length: 4096      # 最大回复长度

  # 截断策略
  truncation: "left"             # left 适用于大多数场景

  # 批次大小
  train_batch_size: 512          # 每次迭代的总批次大小
```

### 2.3 预处理流程

verl 提供了多个数据集的预处理脚本：

1. **GSM8K**（数学推理）
2. **MATH**（高难度数学）
3. **HelloSwag**（常识推理）
4. **Full_hh_rlhf**（人类反馈数据）

#### RLHFDataset 工作流程

```python
# RLHFDataset 自动处理以下步骤：
# 1. 从 Parquet 文件加载数据
# 2. 转换为 HuggingFace Dataset
# 3. 应用 chat_template
# 4. Tokenization
# 5. 过滤超长 prompt（> max_prompt_length）
```

### 2.4 Chat Template 配置

#### 默认行为

```python
# 在 rl_dataset.py 中，默认使用 tokenizer 的 chat template
tokenizer.apply_chat_template(messages, tokenize=True)
```

#### 自定义 Chat Template

**方法 1：修改 tokenizer 配置**

```python
# 修改 tokenizer_config.json
{
  "chat_template": "你的自定义模板"
}
```

**方法 2：使用基座模型（Base Model）**

- verl v0.4+ 支持不使用 chat template
- 适合直接使用基座模型训练

### 2.5 数据准备最佳实践

#### 长度配置策略

```yaml
# 策略 1: 覆盖最长样本
max_prompt_length: 4096  # 足够大以覆盖语料库中最长 prompt

# 策略 2: 截断长尾样本
max_prompt_length: 2048  # 较小值
truncation: "left"       # 配合左截断
```

#### 数据质量控制

1. **过滤策略**：
   - 自动过滤超长样本
   - 确保数据质量

2. **平衡性**：
   - 确保不同难度样本的平衡
   - 避免数据偏斜

3. **格式验证**：
   - 确保所有样本符合 chat_template 格式
   - 验证必需字段完整性

---

## 三、算法关键迭代

### 3.1 PPO (Proximal Policy Optimization)

#### 核心组件

**1. Actor Model（策略模型）**
- 负责生成文本
- 输出 log probabilities

**2. Critic Model（价值函数）**
- 估计状态价值
- 用于计算优势函数（Advantage）

**3. Reference Policy（参考策略）**
- 通常是原始 SFT 模型
- 用于 KL 散度约束

**4. Reward Model（奖励模型）**
- 评估生成文本质量
- 提供奖励信号

#### PPO 迭代流程

```
第 t 次迭代：
1. Rollout 阶段（生成）
   ├─ 使用当前 Actor 生成 N 个样本
   ├─ Reference Policy 计算 log_probs_ref
   └─ Reward Model 评分

2. Advantage 计算
   ├─ Critic 预测价值 V(s)
   ├─ 使用 GAE 计算 Advantage A(s,a)
   └─ GAE = Σ (γλ)^t δ_t，其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

3. Policy Update（策略更新）
   ├─ 计算重要性采样比率 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
   ├─ PPO Clip 目标：
   │  L^CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
   ├─ KL 惩罚（可选）
   └─ 多个 mini-batch 更新

4. Critic Update（价值函数更新）
   └─ 最小化预测误差：L^V = E[(V_θ(s) - V_target)²]
```

#### PPO 配置示例

```yaml
algorithm:
  type: ppo
  gamma: 1.0              # 折扣因子
  lam: 0.95              # GAE lambda
  use_kl_in_reward: True # 在奖励中使用 KL

actor_rollout_ref:
  actor:
    ppo_epochs: 1                    # 每批数据的更新轮数
    ppo_mini_batch_size: 256         # mini-batch 大小
    ppo_micro_batch_size_per_gpu: 4  # micro-batch 大小
    clip_ratio: 0.2                  # 裁剪参数 ε
    use_kl_loss: False               # PPO 不使用额外 KL loss
```

### 3.2 GRPO (Group Relative Policy Optimization)

#### 核心创新

GRPO 简化了 PPO，**消除了 Critic Model 的需求**：

**关键思想**：
- 使用组内平均奖励作为 baseline
- 组内相对比较，无需显式价值函数

#### GRPO 迭代流程

```
第 t 次迭代：
1. Group Sampling（组采样）
   └─ 对每个 prompt 采样 n 次（n > 1，如 n=8）
      prompt_i → [response_1, response_2, ..., response_n]

2. Group Reward 计算
   ├─ 每个 response 获得奖励 r_j
   ├─ 计算组平均奖励：r_mean = (1/n) Σ r_j
   └─ 相对优势：A_j = r_j - r_mean

3. Policy Update（无需 Critic）
   ├─ 计算重要性采样比率 r_t(θ)
   ├─ GRPO 目标：
   │  L^GRPO = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
   │  其中 A_t = r_t - mean(r_group)
   └─ KL 散度约束（作为损失项）
```

#### GRPO 配置示例

```yaml
algorithm:
  type: grpo
  use_kl_in_reward: False  # GRPO 不在奖励中使用 KL

actor_rollout:
  rollout:
    n: 8  # 组采样数量（核心参数，必须 > 1）

actor_rollout_ref:
  actor:
    ppo_epochs: 4              # 多轮更新缓解 rank bias
    ppo_loss_type: "token-mean"  # 推荐 loss 聚合方式
    use_kl_loss: True          # GRPO 使用 KL loss
    kl_loss_coef: 0.001        # KL 系数
    clip_ratio: 0.2            # 裁剪参数
```

#### GRPO 变体：DrGRPO

**Debiased GRPO**：消除长度偏差

```yaml
actor_rollout_ref:
  actor:
    ppo_loss_type: "seq-mean-token-sum-norm"  # DrGRPO 配置
```

### 3.3 DAPO (Decouple Clip and Dynamic sAmpling Policy Optimization)

#### 算法定位

DAPO 是 **2025 年针对长思维链（long-CoT）场景的 SOTA 算法**，特别适合：
- 数学推理（AIME、MATH）
- 代码生成
- 复杂问题求解

#### 核心改进

DAPO = GRPO + 四大创新技术（见 1.1 节）

#### DAPO 迭代流程

```
第 t 次迭代：
1. Dynamic Group Sampling
   ├─ 早期：n = 16（充分探索）
   └─ 后期：n = 8（降低 off-policy）

2. Reward + Shaping
   ├─ 基础奖励（Reward Model）
   ├─ Overlong Reward Shaping
   └─ 无 KL 惩罚（use_kl_in_reward: False）

3. Token-Level Policy Update
   ├─ Token-mean loss 聚合
   ├─ Clip-Higher: clip(r_t, 1-0.2, 1+0.28)
   └─ 相对优势：A_j = r_j - mean(r_group)

4. 梯度更新
   └─ 16 次梯度累积步 × mini-batch size 512
```

#### DAPO 完整配置

```yaml
algorithm:
  type: dapo
  use_kl_in_reward: False     # 关键配置

actor_rollout:
  rollout:
    n: 16  # 动态调整：早期 16，后期可降到 8

actor_rollout_ref:
  actor:
    # Clip-Higher 配置
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28

    # Loss 配置
    ppo_loss_type: "token-mean"
    use_kl_loss: False         # DAPO 不使用 KL loss

    # 批次配置
    ppo_mini_batch_size: 512
    ppo_micro_batch_size_per_gpu: 4
    gradient_accumulation_steps: 16

    # 优化配置
    ppo_epochs: 1

optimizer:
  type: AdamW
  lr: 1e-6
  warmup_steps: 20
```

### 3.4 算法对比总结

| 特性 | PPO | GRPO | DAPO |
|-----|-----|------|------|
| **是否需要 Critic** | ✅ 需要 | ❌ 不需要 | ❌ 不需要 |
| **Baseline** | Critic V(s) | 组平均奖励 | 组平均奖励 |
| **采样策略** | 每个 prompt 1 次 | 每个 prompt n 次 | 动态采样 |
| **KL 约束位置** | 奖励中 | 损失中 | 无直接 KL |
| **Clip 策略** | 对称 [1-ε,1+ε] | 对称 [1-ε,1+ε] | 非对称 Clip-Higher |
| **Loss 聚合** | - | token-mean/seq-mean | token-mean（推荐） |
| **适用场景** | 通用 RLHF | 通用 RLHF | 长 CoT 推理 |
| **计算成本** | 高（需训练 Critic） | 中（多次采样） | 中（多次采样） |
| **稳定性** | 中 | 中（存在 rank bias） | 高（针对性优化） |

### 3.5 训练收敛监控

#### 关键指标

```yaml
# 监控以下指标判断训练健康度：

1. 奖励指标
   - reward/mean: 平均奖励（应持续上升）
   - reward/std: 奖励标准差（不应过小，避免模式坍塌）

2. Policy 指标
   - policy/entropy: 策略熵（不应快速下降）
   - policy/kl: KL 散度（应保持在合理范围）
   - policy/clip_ratio: 裁剪比率（~10-30% 被裁剪为健康）

3. Loss 指标
   - loss/policy: 策略损失
   - loss/value: 价值损失（仅 PPO）
   - loss/kl: KL 损失（GRPO）

4. 性能指标
   - throughput/samples_per_second
   - time/rollout_time
   - time/update_time
```

#### 常见问题诊断

**问题 1：Entropy Collapse（熵坍塌）**
```
症状：policy/entropy 快速下降到接近 0
解决：使用 Clip-Higher（DAPO）或增加 entropy_coef
```

**问题 2：Reward Hacking（奖励黑客）**
```
症状：reward 快速上升但实际质量下降，KL 散度爆炸
解决：增大 kl_loss_coef 或降低学习率
```

**问题 3：训练不稳定**
```
症状：Loss 震荡，reward 波动大
解决：降低学习率，增加 ppo_epochs，使用 gradient clipping
```

---

## 四、架构设计

### 4.1 HybridFlow 核心架构

HybridFlow 由三大核心组件构成：

#### 1. Hybrid Programming Model（混合编程模型）

**核心思想**：分离控制流和计算流

```python
# 伪代码示例
class HybridFlowController:
    def run_iteration(self):
        # 控制流：定义 RL 算法逻辑
        prompts = self.get_batch()

        # 计算流：传递给 Worker 执行
        responses = self.rollout_worker.generate(prompts)
        rewards = self.reward_worker.compute(responses)

        # 控制流：继续算法逻辑
        advantages = self.compute_advantages(rewards)
        self.actor_worker.update(advantages)
```

**优势**：
- 算法逻辑与计算引擎解耦
- 轻松替换不同的训练/推理后端
- 灵活组合不同的 Worker

#### 2. 3D-HybridEngine

**问题**：Actor 模型在训练和生成阶段需要不同的并行配置
- 训练阶段：FSDP（全切分并行）适合梯度计算
- 生成阶段：TP（张量并行）适合推理吞吐

**解决方案**：3D-HybridEngine 支持动态切换

```yaml
# Actor 训练配置（FSDP）
fsdp_config:
  sharding_strategy: "full_shard"  # FSDP

# Rollout 生成配置（vLLM with TP）
rollout_config:
  tensor_model_parallel_size: 4    # TP=4
```

**技术特点**：
- **零内存冗余**：训练到生成转换无需重复加载权重
- **最小化通信开销**：高效的权重重分布（resharding）
- **灵活设备映射**：适应不同集群规模

#### 3. Auto-Mapping Algorithm（自动映射算法）

自动决定最优的：
- Worker 放置策略
- 并行配置
- 设备分配

### 4.2 Worker 角色体系

verl 定义了多种 Worker 角色：

```
┌─────────────────────────────────────────┐
│          Controller (控制流)             │
└─────────────────────────────────────────┘
         │
         ├─────────────┬─────────────┬──────────────┐
         │             │             │              │
    ┌────▼───┐   ┌────▼───┐   ┌─────▼────┐   ┌────▼────┐
    │ Actor  │   │Rollout │   │  Critic  │   │ Reward  │
    │ Worker │   │ Worker │   │  Worker  │   │  Model  │
    └────────┘   └────────┘   └──────────┘   └─────────┘
    (FSDP/     (vLLM/SGLang)   (FSDP/        (FSDP/
     Megatron)                  Megatron)     Megatron)
```

#### Worker 类型说明

**1. Actor Worker**
- 策略模型训练
- 支持 FSDP/Megatron-LM 后端

**2. Rollout Worker**
- 文本生成
- 支持 vLLM/SGLang/HF Transformers

**3. ActorRollout Worker (HybridEngine)**
- Actor + Rollout 融合
- 消除权重同步开销

**4. Critic Worker**
- 价值函数训练（仅 PPO）
- 支持 FSDP/Megatron-LM

**5. Reward Model Worker**
- 奖励计算
- 支持外部 RM 或基于函数的奖励

**6. Reference Policy Worker**
- 冻结的参考策略
- 用于 KL 散度计算

**7. ActorRolloutRef Worker**
- Actor + Rollout + Reference 三合一
- 最大化内存效率

### 4.3 训练后端对比

#### FSDP (Fully Sharded Data Parallel)

```yaml
fsdp_config:
  sharding_strategy: "full_shard"  # 或 "shard_grad_op"
  forward_prefetch: True           # 性能优化
  backward_prefetch: "backward_pre"
  cpu_offload: False
```

**特点**：
- PyTorch 原生支持
- 易于使用和调试
- FSDP2 进一步优化性能

**适用场景**：
- 中小规模集群（<100 GPUs）
- 快速原型开发

#### Megatron-LM

```yaml
megatron_config:
  tensor_model_parallel_size: 8      # TP
  pipeline_model_parallel_size: 4    # PP
  expert_model_parallel_size: 2      # EP（专家并行）
```

**特点**：
- 支持超大规模模型（671B+）
- 3D 并行（TP + PP + DP）
- 专家并行支持（MoE 模型）

**适用场景**：
- 大规模集群（100+ GPUs）
- 超大模型（>100B）

### 4.4 推理后端对比

#### vLLM

```yaml
rollout_config:
  backend: vllm
  tensor_model_parallel_size: 4
  gpu_memory_utilization: 0.6        # 关键参数
  max_num_batched_tokens: 4096
```

**特点**：
- PagedAttention 技术
- 高吞吐量
- 推荐版本：v0.8.2+

**性能提示**：
- `gpu_memory_utilization` 推荐 0.5-0.7
- 优先使用 DP（数据并行）而非 TP

#### SGLang

```yaml
rollout_config:
  backend: sglang
```

**特点**：
- 支持多轮对话
- 工具调用（Tool Calling）
- 结构化输出

---

## 五、性能优化

### 5.1 关键性能参数

#### GPU 内存利用率

```yaml
rollout_config:
  gpu_memory_utilization: 0.6  # 推荐范围: 0.5-0.7
```

**调优建议**：
- 太低（<0.5）：吞吐量不足
- 太高（>0.8）：容易 OOM
- 最佳：0.6 平衡性能和稳定性

#### 并行策略

**原则**：DP（数据并行）> TP（张量并行）

```yaml
# 示例：8 GPU 配置
tensor_model_parallel_size: 2   # TP=2
# 自动产生 4 个 vLLM 副本（DP=4）
```

**原因**：
- DP 可并行处理不同 batch
- TP 需要 GPU 间通信
- 在 GPU 资源充足时，DP 吞吐更高

#### FSDP Forward Prefetch

```yaml
fsdp_config:
  forward_prefetch: True  # 开启预取
```

**效果**：
- 通信与计算重叠
- 减少 GPU 空闲时间
- 显著提升训练吞吐

### 5.2 批次大小调优

```yaml
# 分层批次配置
data:
  train_batch_size: 512              # 总批次（prompt 数）

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 512         # mini-batch（≤ train_batch_size * n）
    ppo_micro_batch_size_per_gpu: 4  # micro-batch（单 GPU 前向传播）
    gradient_accumulation_steps: 16  # 梯度累积步数
```

**调优策略**：
1. **train_batch_size**：
   - 较大值：减少 rollout 次数，但增加 off-policy 偏差
   - 较小值：更频繁更新，但增加开销
   - 推荐：256-512

2. **ppo_mini_batch_size**：
   - 应 ≤ train_batch_size × n
   - 决定每次更新的样本数

3. **ppo_micro_batch_size_per_gpu**：
   - 尽可能大（受显存限制）
   - 直接影响 GPU 利用率

4. **gradient_accumulation_steps**：
   - 等效更大的 mini_batch_size
   - 平衡显存和收敛速度

### 5.3 长度配置优化

```yaml
data:
  max_prompt_length: 2048
  max_response_length: 4096
  truncation: "left"  # 推荐左截断
```

**策略**：
- **max_prompt_length**：覆盖大部分样本（如 95%）即可
- **truncation: left**：保留最新的上下文信息
- 过滤超长样本避免拖慢训练

### 5.4 实验追踪

verl 支持多种实验追踪工具：

```yaml
# wandb
logger:
  type: wandb
  project: my-rlhf-project
  name: experiment-1

# mlflow
logger:
  type: mlflow
  tracking_uri: http://localhost:5000

# tensorboard
logger:
  type: tensorboard
  log_dir: ./logs

# swanlab
logger:
  type: swanlab
```

### 5.5 性能 Profiling

#### 启用 Profiling

```yaml
profiling:
  enabled: True
  output_dir: ./profiles
```

#### 关键指标监控

```
1. Throughput（吞吐量）
   - samples/second
   - tokens/second

2. Time Breakdown（时间分解）
   - rollout_time（生成时间）
   - update_time（训练时间）
   - data_loading_time

3. Memory（显存）
   - peak_memory_allocated
   - memory_utilization

4. Communication（通信）
   - allreduce_time
   - allgather_time
```

### 5.6 推荐配置模板

#### 配置 1：中小规模（8-32 GPUs，7B-13B 模型）

```yaml
# 训练后端
trainer: fsdp
fsdp_config:
  forward_prefetch: True

# 推理后端
rollout:
  backend: vllm
  gpu_memory_utilization: 0.6
  tensor_model_parallel_size: 2

# 批次配置
data:
  train_batch_size: 256
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 8
```

#### 配置 2：大规模（100+ GPUs，70B+ 模型）

```yaml
# 训练后端
trainer: megatron
megatron_config:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 4

# 推理后端
rollout:
  backend: vllm
  gpu_memory_utilization: 0.6
  tensor_model_parallel_size: 8

# 批次配置
data:
  train_batch_size: 512
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 512
    ppo_micro_batch_size_per_gpu: 4
    gradient_accumulation_steps: 16
```

---

## 附录

### A. 支持的模型

- **Llama 系列**：Llama3, Llama3.1, Llama3.2
- **Qwen 系列**：Qwen2, Qwen2.5, Qwen3
- **其他**：Gemma2, DeepSeek-LLM, Mistral, GLM-4

### B. 多模态支持

verl 支持视觉-语言模型（VLM）的 RLHF 训练：
- Qwen2-VL
- LLaVA 系列

### C. 社区资源

- **文档**：https://verl.readthedocs.io
- **GitHub**：https://github.com/volcengine/verl
- **论文**：HybridFlow (EuroSys 2025)
- **社区项目**：
  - DeepSeek-R1 复现
  - Qwen 推理模型
  - 数学/代码 Agent

### D. 安装指南

```bash
# 基础安装
pip install verl

# 推荐：使用 Docker
docker pull hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2-verl0.3.0.post1

# 从源码安装
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

### E. 常见问题 FAQ

**Q1：PPO 和 GRPO 如何选择？**
- 小规模实验：GRPO（更简单，无需 Critic）
- 稳定性要求高：PPO
- 长 CoT 场景：DAPO

**Q2：如何处理 OOM？**
- 降低 `ppo_micro_batch_size_per_gpu`
- 增加 `gradient_accumulation_steps`
- 降低 `max_response_length`
- 使用 `gradient_checkpointing`

**Q3：训练不收敛怎么办？**
- 检查数据质量
- 降低学习率
- 检查 KL 系数设置
- 监控 entropy 和 reward 曲线

---

## 总结

verl 作为 2025 年最先进的 LLM RLHF 训练框架，具有以下核心优势：

1. **灵活性**：支持多种 RL 算法和训练范式
2. **高性能**：3D-HybridEngine 实现 SOTA 吞吐量
3. **易用性**：混合编程模型简化算法实现
4. **可扩展性**：从小规模到超大规模无缝扩展
5. **生产就绪**：经过字节跳动大规模生产验证

特别是 **DAPO 算法**在长思维链场景的突破，为 LLM 推理能力提升提供了强有力的工具。

对于算法工程师而言，理解 verl 的三大支柱——**训练 Tricks**、**数据准备**、**算法迭代**——是充分发挥其威力的关键。

---

**文档版本**: v1.0
**最后更新**: 2025-11-18
**基于 verl 版本**: v0.3.0+
**参考资料**: GitHub, 官方文档, HybridFlow 论文, DAPO 论文
