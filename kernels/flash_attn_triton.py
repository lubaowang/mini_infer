"""
kernels/flash_attn_triton.py  （修正版）
────────────────────────────
Triton Flash Attention（Causal），用于 prefill 阶段。

修正点（相对原版）：
  1. 写回 dtype 判断：原版 `Out_ptr.dtype.element_ty` 在某些 triton 版本
     对 tl.pointer_type 不可用；改为在 Python 侧根据 out.dtype 决定 cast 目标，
     用 .to(tl.float16) / .to(tl.bfloat16) / .to(tl.float32) 条件分支。
     最简处理：kernel 内全程 fp32 计算，写回时用 tl.cast 到调用方指定的 dtype。
  2. acc / exp_s 的 dtype 保持一致，dot 乘法结果显式 cast 到 fp32。
  3. causal tile 跳过逻辑：原版 `if CAUSAL: tile_valid = ...` 在 Triton 中
     constexpr if 内不能有动态比较结果赋值到 mask，改为直接把 tile_valid 折入 mask_kv。
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, Hq, Hk, T_q, T_kv, D,
    scale,
    kv_group_size,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_D:  tl.constexpr,
    CAUSAL:   tl.constexpr,
    OUT_FP16: tl.constexpr,   # ✅ 新增：由 Python 层告知写回 dtype
    OUT_BF16: tl.constexpr,
):
    bh_id  = tl.program_id(0)
    tile_m = tl.program_id(1)

    b  = bh_id // Hq
    hq = bh_id  % Hq
    hk = hq // kv_group_size

    q_start = tile_m * BLOCK_M
    q_offs  = q_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)
    mask_qm = q_offs < T_q
    mask_d  = d_offs < D

    Q_base = Q_ptr + b * stride_qb + hq * stride_qh
    q = tl.load(
        Q_base + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
        mask=mask_qm[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32) * scale

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    K_base = K_ptr + b * stride_kb + hk * stride_kh
    V_base = V_ptr + b * stride_vb + hk * stride_vh

    n_tiles_kv = tl.cdiv(T_kv, BLOCK_N)
    for tile_n in range(n_tiles_kv):
        kv_start = tile_n * BLOCK_N
        kv_offs  = kv_start + tl.arange(0, BLOCK_N)
        mask_kv  = kv_offs < T_kv

        # ✅ causal：整块 tile 在 query 之后则全部 mask 掉（不 break，避免 Triton 控制流问题）
        if CAUSAL:
            tile_valid = kv_start <= (q_start + BLOCK_M - 1)
            mask_kv = mask_kv & tile_valid

        k = tl.load(
            K_base + kv_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # ✅ dot 结果显式 cast fp32
        scores = tl.dot(q, tl.trans(k)).to(tl.float32)

        if CAUSAL:
            causal_mask = q_offs[:, None] >= kv_offs[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        scores = tl.where(mask_kv[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        exp_s = tl.exp(scores - m_new[:, None])
        sum_exp = tl.sum(exp_s, axis=1)
        l_new = tl.exp(m_i - m_new) * l_i + sum_exp + 1e-8  # ✅ eps 防除零

        v = tl.load(
            V_base + kv_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        rescale = (tl.exp(m_i - m_new) * l_i / l_new)[:, None]
        # ✅ exp_s cast 到 fp32 再 dot
        acc = acc * rescale + tl.dot(exp_s.to(tl.float32), v) / l_new[:, None]

        m_i = m_new
        l_i = l_new

    Out_base = Out_ptr + b * stride_ob + hq * stride_oh
    # ✅ 写回 dtype：由 constexpr 参数控制，避免 Out_ptr.dtype.element_ty 访问问题
    if OUT_FP16:
        out_val = acc.to(tl.float16)
    elif OUT_BF16:
        out_val = acc.to(tl.bfloat16)
    else:
        out_val = acc

    tl.store(
        Out_base + q_offs[:, None] * stride_ot + d_offs[None, :] * stride_od,
        out_val,
        mask=mask_qm[:, None] & mask_d[None, :],
    )


def flash_attn_triton(
    q: torch.Tensor,    # (B, Hq, T_q, D)
    k: torch.Tensor,    # (B, Hk, T_kv, D)
    v: torch.Tensor,    # (B, Hk, T_kv, D)
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    B, Hq, T_q, D = q.shape
    _, Hk, T_kv, _ = k.shape
    assert Hq % Hk == 0, "GQA 要求 Hq 整除 Hk"
    kv_group_size = Hq // Hk

    if scale is None:
        scale = D ** -0.5

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_M = 64
    BLOCK_N = 64

    out = torch.empty_like(q)

    # ✅ Python 侧确定写回 dtype，作为 constexpr 传入
    out_fp16 = (out.dtype == torch.float16)
    out_bf16 = (out.dtype == torch.bfloat16)

    grid = (B * Hq, triton.cdiv(T_q, BLOCK_M))
    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, Hq, Hk, T_q, T_kv, D,
        scale, kv_group_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        CAUSAL=causal,
        OUT_FP16=out_fp16,
        OUT_BF16=out_bf16,
    )
    return out
