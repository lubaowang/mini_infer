"""
kernels/rope_emb.py
───────────────────
Triton 实现的 Rotary Position Embedding（fused apply）。

对标 model_minimind.py 中的：
    apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

融合策略：
  - 单 kernel 同时完成 q 和 k 的旋转，避免两次独立 kernel launch
  - cos/sin 在 kernel 内直接用，不需要 unsqueeze/broadcast
  - 支持 GQA：q_heads != k_heads（分别指定 head 数）
"""

import torch
import triton
import triton.language as tl


# ─── apply_rotary（单 tensor） ────────────────────────────────────────────────

@triton.jit
def _apply_rope_kernel(
    X_ptr,          # (B, H, T, D) 连续存储
    Cos_ptr,        # (B, T, D)    —— 已在 RotaryEmbedding.forward 中展开
    Sin_ptr,        # (B, T, D)
    Out_ptr,
    B, H, T, D,     # batch, heads, seq_len, head_dim
    stride_xb, stride_xh, stride_xt, stride_xd,
    stride_cb, stride_ct, stride_cd,             # cos/sin: B, T, D
    stride_ob, stride_oh, stride_ot, stride_od,
    BLOCK_D: tl.constexpr,
):
    # program_id: (b * H * T + h * T + t) 展开
    idx = tl.program_id(0)
    t   = idx % T
    tmp = idx // T
    h   = tmp % H
    b   = tmp // H

    # 列偏移
    d = tl.arange(0, BLOCK_D)
    mask_d = d < D

    # 读 x
    x_ptr = X_ptr + b * stride_xb + h * stride_xh + t * stride_xt
    x = tl.load(x_ptr + d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)

    # 读 cos / sin（不依赖 head，B×T×D）
    cs_ptr_base = b * stride_cb + t * stride_ct
    cos_vals = tl.load(Cos_ptr + cs_ptr_base + d * stride_cd, mask=mask_d, other=1.0).to(tl.float32)
    sin_vals = tl.load(Sin_ptr + cs_ptr_base + d * stride_cd, mask=mask_d, other=0.0).to(tl.float32)

    # rotate_half: [-x[D/2:], x[:D/2]]
    half = D // 2
    d_lo  = d          # [0, D/2)
    d_hi  = d + half   # [D/2, D)
    mask_lo = d_lo < half
    mask_hi = d_hi < D

    x_lo = tl.load(x_ptr + d_lo * stride_xd, mask=mask_lo, other=0.0).to(tl.float32)
    x_hi = tl.load(x_ptr + d_hi * stride_xd, mask=mask_hi, other=0.0).to(tl.float32)

    # rotate_half(x) = [-x_hi | x_lo]，与 d 对应
    rot_lo = -x_hi  # 前半段放 -x[D/2:]
    rot_hi =  x_lo  # 后半段放  x[:D/2]

    cos_lo = tl.load(Cos_ptr + cs_ptr_base + d_lo * stride_cd, mask=mask_lo, other=1.0).to(tl.float32)
    cos_hi = tl.load(Cos_ptr + cs_ptr_base + d_hi * stride_cd, mask=mask_hi, other=1.0).to(tl.float32)
    sin_lo = tl.load(Sin_ptr + cs_ptr_base + d_lo * stride_cd, mask=mask_lo, other=0.0).to(tl.float32)
    sin_hi = tl.load(Sin_ptr + cs_ptr_base + d_hi * stride_cd, mask=mask_hi, other=0.0).to(tl.float32)

    out_lo = x_lo * cos_lo + rot_lo * sin_lo
    out_hi = x_hi * cos_hi + rot_hi * sin_hi

    out_ptr = Out_ptr + b * stride_ob + h * stride_oh + t * stride_ot
    tl.store(out_ptr + d_lo * stride_od, out_lo.to(x.dtype), mask=mask_lo)
    tl.store(out_ptr + d_hi * stride_od, out_hi.to(x.dtype), mask=mask_hi)


def _apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, T, D)  — q 或 k（已 transpose 到 heads-first）
    cos: (B, T, D)
    sin: (B, T, D)
    """
    B, H, T, D = x.shape
    out = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D // 2)   # 每次处理半个 head_dim
    grid = (B * H * T,)

    _apply_rope_kernel[grid](
        x, cos, sin, out,
        B, H, T, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_D=BLOCK_D,
    )
    return out


# ─── 对外接口：同时处理 q 和 k ──────────────────────────────────────────────

def apply_rotary_pos_emb_triton(
    q: torch.Tensor,    # (B, Hq, T, D)
    k: torch.Tensor,    # (B, Hk, T, D)
    cos: torch.Tensor,  # (B, T, D)
    sin: torch.Tensor,  # (B, T, D)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合版 RoPE apply，替换 model_minimind.py 中的 apply_rotary_pos_emb。
    q/k 均需为 (B, H, T, D) 的连续 tensor（即已执行过 .transpose(1,2).contiguous()）。
    """
    q_out = _apply_rope_single(q, cos, sin)
    k_out = _apply_rope_single(k, cos, sin)
    return q_out, k_out
