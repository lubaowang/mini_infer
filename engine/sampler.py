"""
engine/sampler.py
─────────────────
解耦的采样器，支持：
  - Greedy（argmax）
  - Temperature scaling
  - Top-P（nucleus sampling）
  - Temperature + Top-P 组合

对标原 eval.py 中 model.generate() 的 do_sample=True, top_p, temperature 参数。
手动实现便于后续在 Triton 中进一步融合。
"""

import torch
import torch.nn.functional as F
from typing import Optional


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: (B, vocab_size)
    Returns:
        token_ids: (B,)
    """
    return logits.argmax(dim=-1)


def temperature_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """纯 temperature 采样（不截断）。"""
    if temperature <= 0.0:
        return greedy_sample(logits)
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def top_p_sample(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Nucleus（Top-P）采样。

    算法：
      1. 对 logits 按概率降序排列
      2. 计算累积概率，截断超过 top_p 的 tail
      3. 在保留的 token 上重新 softmax 采样

    Args:
        logits:             (B, vocab_size)
        top_p:              截断阈值，如 0.85
        temperature:        缩放温度（先 scale 再截断）
        min_tokens_to_keep: 至少保留的候选 token 数（防止退化）

    Returns:
        token_ids: (B,)
    """
    if temperature > 0.0:
        logits = logits / temperature

    # 降序排列
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 移除累积概率超过 top_p 的 token（保留刚好超过阈值前的所有 token）
    # shift right：第 i 个 token 的 cumsum 包含自身，所以右移一位再比较
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    # 至少保留 min_tokens_to_keep 个
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # 用 -inf 填充被移除的 token，然后在原始顺序上 scatter 回去
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    统一采样入口。

    Args:
        logits:      (B, vocab_size)  或  (B, 1, vocab_size)
        temperature: 生成温度
        top_p:       nucleus 阈值（1.0 = 不截断）
        do_sample:   False 时走 greedy

    Returns:
        next_token_ids: (B,)
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]   # 取最后一个 token 的 logits

    if not do_sample or temperature == 0.0:
        return greedy_sample(logits)

    if top_p < 1.0:
        return top_p_sample(logits, top_p=top_p, temperature=temperature)

    return temperature_sample(logits, temperature=temperature)
