"""
线性注意力机制实现集合

提供多种线性注意力机制的基本实现：
- Standard Attention (SA)
- Mamba2
- Gated Linear Attention (GLA)
- DeltaNet
- Gated Delta Network (GDN)
- Key-Decomposed Attention (KDA)
"""

from .standard_attention import StandardAttention, CausalStandardAttention
from .mamba2 import Mamba2, Mamba2Block
from .gated_linear_attention import GatedLinearAttention, ParallelGLA
from .deltanet import DeltaNet, ParallelDeltaNet
from .gated_delta_network import GatedDeltaNetwork, HierarchicalGDN
from .key_decomposed_attention import KeyDecomposedAttention, AdaptiveKDA

__version__ = "0.1.0"

__all__ = [
    # Standard Attention
    "StandardAttention",
    "CausalStandardAttention",

    # Mamba2
    "Mamba2",
    "Mamba2Block",

    # GLA
    "GatedLinearAttention",
    "ParallelGLA",

    # DeltaNet
    "DeltaNet",
    "ParallelDeltaNet",

    # GDN
    "GatedDeltaNetwork",
    "HierarchicalGDN",

    # KDA
    "KeyDecomposedAttention",
    "AdaptiveKDA",
]


def get_attention_module(name, d_model, num_heads=8, **kwargs):
    """
    工厂函数：根据名称获取注意力模块

    Args:
        name: 注意力类型名称
        d_model: 模型维度
        num_heads: 注意力头数
        **kwargs: 其他参数

    Returns:
        注意力模块实例

    Example:
        >>> attn = get_attention_module('gla', d_model=512, num_heads=8)
        >>> output = attn(x)
    """
    name = name.lower()

    if name in ['sa', 'standard', 'standard_attention']:
        return StandardAttention(d_model, num_heads, **kwargs)

    elif name in ['mamba', 'mamba2']:
        return Mamba2(d_model, **kwargs)

    elif name in ['gla', 'gated_linear_attention']:
        return GatedLinearAttention(d_model, num_heads, **kwargs)

    elif name in ['deltanet', 'delta']:
        return DeltaNet(d_model, num_heads, **kwargs)

    elif name in ['gdn', 'gated_delta_network']:
        return GatedDeltaNetwork(d_model, num_heads, **kwargs)

    elif name in ['kda', 'key_decomposed_attention']:
        return KeyDecomposedAttention(d_model, num_heads, **kwargs)

    else:
        raise ValueError(
            f"Unknown attention type: {name}. "
            f"Available: sa, mamba2, gla, deltanet, gdn, kda"
        )


def list_available_attentions():
    """列出所有可用的注意力机制"""
    attentions = {
        'standard_attention': {
            'aliases': ['sa', 'standard'],
            'description': '标准的 softmax 注意力',
            'complexity': 'O(T²D)',
            'use_case': '短序列，需要最强表达能力'
        },
        'mamba2': {
            'aliases': ['mamba'],
            'description': '基于状态空间模型的选择性 SSM',
            'complexity': 'O(TDN)',
            'use_case': '超长序列，需要极致效率'
        },
        'gla': {
            'aliases': ['gated_linear_attention'],
            'description': '门控线性注意力',
            'complexity': 'O(TD²)',
            'use_case': '中长序列，平衡性能和实现'
        },
        'deltanet': {
            'aliases': ['delta'],
            'description': '距离感知的线性注意力',
            'complexity': 'O(TD²)',
            'use_case': '位置敏感的任务'
        },
        'gdn': {
            'aliases': ['gated_delta_network'],
            'description': '多重门控 + 自适应衰减',
            'complexity': 'O(TD²)',
            'use_case': '复杂任务，需要强表达能力'
        },
        'kda': {
            'aliases': ['key_decomposed_attention'],
            'description': '键分解注意力',
            'complexity': 'O(T²D/n) - O(TD²/n)',
            'use_case': '需要灵活性，多模态任务'
        }
    }

    return attentions


if __name__ == "__main__":
    print("线性注意力机制实现库")
    print("=" * 60)

    attentions = list_available_attentions()

    for name, info in attentions.items():
        print(f"\n{name.upper()}")
        print(f"  别名: {', '.join(info['aliases'])}")
        print(f"  描述: {info['description']}")
        print(f"  复杂度: {info['complexity']}")
        print(f"  适用: {info['use_case']}")

    print("\n" + "=" * 60)
    print("使用示例：")
    print("  from linear_attention import get_attention_module")
    print("  attn = get_attention_module('gla', d_model=512, num_heads=8)")
    print("  output = attn(x)")
