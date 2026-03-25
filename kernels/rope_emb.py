"""
kernels/rope_emb.py  （修正版）
───────────────────
Triton 实现的 Rotary Position Embedding（fused apply）。

修正点（相对原版）：
  1. BLOCK_D 错误：原版 BLOCK_D = next_pow2(D//2)，但 kernel 内同时处理
     d_lo=[0,D/2) 和 d_hi=[D/2,D)，每次实际访问 D 个元素，BLOCK_D 需 >= D//2
     且 lo/hi 各自需要 D//2 个 lane。修正为两个独立 tile，每个 BLOCK_HALF = D//2。
  2. kernel 内 x/cos/sin 被读了两次（d 和 d_lo/d_hi 两组），去掉第一次无用读取。
  3. stride 传参统一，确保 out 和 x 使用相同 stride。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rope_kernel(
    X_ptr,
    Cos_ptr,
    Sin_ptr,
    Out_ptr,
    B, H, T, D,
    stride_xb, stride_xh, stride_xt, stride_xd,
    stride_cb, stride_ct, stride_cd,
    stride_ob, stride_oh, stride_ot, stride_od,
    BLOCK_HALF: tl.constexpr,   # = D // 2，每个 program 处理半个 head_dim
):
    """
    每个 program 处理一个 (b, h, t) 位置的整个 head_dim（分前后两半）。
    前半 d_lo = [0, D/2)，后半 d_hi = [D/2, D)。
    rotate_half(x) = [-x[D/2:], x[:D/2]]
    因此：
      out_lo = x_lo * cos_lo + (-x_hi) * sin_lo
      out_hi = x_hi * cos_hi +   x_lo  * sin_hi
    """
    idx = tl.program_id(0)
    t   = idx % T
    tmp = idx // T
    h   = tmp % H
    b   = tmp // H

    # 前半索引 [0, BLOCK_HALF)
    d_lo   = tl.arange(0, BLOCK_HALF)           # [0, D/2)
    mask_lo = d_lo < (D // 2)

    # 后半索引 [D/2, D)
    d_hi   = d_lo + (D // 2)                    # [D/2, D)
    mask_hi = d_hi < D

    # 基址
    x_base   = X_ptr   + b * stride_xb + h * stride_xh + t * stride_xt
    cos_base = Cos_ptr + b * stride_cb             + t * stride_ct
    sin_base = Sin_ptr + b * stride_cb             + t * stride_ct
    out_base = Out_ptr + b * stride_ob + h * stride_oh + t * stride_ot

    # 读取前后两半 x
    x_lo = tl.load(x_base + d_lo * stride_xd, mask=mask_lo, other=0.0).to(tl.float32)
    x_hi = tl.load(x_base + d_hi * stride_xd, mask=mask_hi, other=0.0).to(tl.float32)

    # 读取 cos / sin（cos/sin 不依赖 head，对所有 head 相同）
    cos_lo = tl.load(cos_base + d_lo * stride_cd, mask=mask_lo, other=1.0).to(tl.float32)
    cos_hi = tl.load(cos_base + d_hi * stride_cd, mask=mask_hi, other=1.0).to(tl.float32)
    sin_lo = tl.load(sin_base + d_lo * stride_cd, mask=mask_lo, other=0.0).to(tl.float32)
    sin_hi = tl.load(sin_base + d_hi * stride_cd, mask=mask_hi, other=0.0).to(tl.float32)

    # rotate_half: 前半 = -x_hi，后半 = x_lo
    out_lo = x_lo * cos_lo + (-x_hi) * sin_lo
    out_hi = x_hi * cos_hi +   x_lo  * sin_hi

    # 写回
    tl.store(out_base + d_lo * stride_od, out_lo.to(x_lo.dtype), mask=mask_lo)
    tl.store(out_base + d_hi * stride_od, out_hi.to(x_hi.dtype), mask=mask_hi)


def _apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, T, D)  — 连续存储
    cos: (B, T, D)
    sin: (B, T, D)
    """
    B, H, T, D = x.shape
    assert D % 2 == 0, "head_dim 必须是偶数"
    out = torch.empty_like(x)

    BLOCK_HALF = triton.next_power_of_2(D // 2)

    _apply_rope_kernel[(B * H * T,)](
        x, cos, sin, out,
        B, H, T, D,
        x.stride(0),   x.stride(1),   x.stride(2),   x.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_HALF=BLOCK_HALF,
    )
    return out


def apply_rotary_pos_emb_triton(
    q: torch.Tensor,    # (B, Hq, T, D)  — 已 transpose + contiguous
    k: torch.Tensor,    # (B, Hk, T, D)
    cos: torch.Tensor,  # (B, T, D)
    sin: torch.Tensor,  # (B, T, D)
) -> tuple:
    """
    替换 model_minimind.apply_rotary_pos_emb。
    q/k 需为 heads-first 且内存连续。
    """
    q_out = _apply_rope_single(q, cos, sin)
    k_out = _apply_rope_single(k, cos, sin)
    return q_out, k_out
