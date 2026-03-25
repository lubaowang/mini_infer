"""
kernels/flash_attn_triton.py
────────────────────────────
手写 Triton Flash Attention（Causal），对标 MiniMind Attention.forward 中的：

    F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

实现基于 Tri Dao / Phil Tillet 的 Flash Attention 2 算法：
  - Online softmax（Softmax-Tiled），单次遍历 K/V，O(seq) SRAM 占用
  - Causal mask：tile 级别跳过，右上角 tile 不参与计算
  - 支持 GQA：通过 kv_group_size 参数，k/v head 数可小于 q head 数

适用场景：prefill 阶段（seq_len > 1）
decode 阶段（seq_len == 1）退化为 GEMV，由 gemv.py 接管。
"""

import torch
import triton
import triton.language as tl
import math


# ─── 核心 Triton kernel ───────────────────────────────────────────────────────

@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # strides: (batch, head, seq, dim)
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, Hq, Hk, T_q, T_kv, D,
    scale,                          # 1 / sqrt(head_dim)
    kv_group_size,                  # Hq // Hk（GQA 分组数）
    BLOCK_M: tl.constexpr,          # query tile 大小（沿 T_q 方向）
    BLOCK_N: tl.constexpr,          # key/value tile 大小（沿 T_kv 方向）
    BLOCK_D: tl.constexpr,          # head_dim tile（必须 >= D）
    CAUSAL:  tl.constexpr,          # 是否施加 causal mask
):
    # program: (batch * Hq, tile_m) 两级
    bh_id  = tl.program_id(0)   # batch * Hq 索引
    tile_m = tl.program_id(1)   # query tile 编号

    b  = bh_id // Hq
    hq = bh_id  % Hq
    hk = hq // kv_group_size    # GQA：找对应的 kv head

    # Q tile 行偏移
    q_start = tile_m * BLOCK_M
    q_offs  = q_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)
    mask_qm = q_offs < T_q
    mask_d  = d_offs < D

    # 读入 Q tile: (BLOCK_M, D)
    Q_base = Q_ptr + b * stride_qb + hq * stride_qh
    q = tl.load(
        Q_base + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
        mask=mask_qm[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32) * scale

    # ── Online softmax accumulators ──
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)               # row sum(exp)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)      # weighted value

    K_base = K_ptr + b * stride_kb + hk * stride_kh
    V_base = V_ptr + b * stride_vb + hk * stride_vh

    # KV 方向遍历
    n_tiles_kv = tl.cdiv(T_kv, BLOCK_N)
    for tile_n in range(n_tiles_kv):
        kv_start = tile_n * BLOCK_N
        kv_offs  = kv_start + tl.arange(0, BLOCK_N)

        # base mask（padding）
        mask_kv = kv_offs < T_kv

        # === ✅ 替代 break：整块 tile 是否有效 ===
        if CAUSAL:
            tile_valid = kv_start <= (q_start + BLOCK_M - 1)
            mask_kv = mask_kv & tile_valid

        # ---------------- K ----------------
        k = tl.load(
            K_base + kv_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # scores: (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, tl.trans(k))

        # === causal mask（tile 内精细）===
        if CAUSAL:
            causal_mask = q_offs[:, None] >= kv_offs[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # padding mask
        scores = tl.where(mask_kv[None, :], scores, float("-inf"))

        # ---------------- online softmax ----------------
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        exp_s = tl.exp(scores - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_s, axis=1)

        # ---------------- V ----------------
        v = tl.load(
            V_base + kv_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # ---------------- acc update ----------------
        acc = acc * (tl.exp(m_i - m_new) * l_i / l_new)[:, None] \
            + tl.dot(exp_s.to(v.dtype), v) / l_new[:, None]

        m_i = m_new
        l_i = l_new

    # 写回输出: (BLOCK_M, D)
    Out_base = Out_ptr + b * stride_ob + hq * stride_oh
    tl.store(
        Out_base + q_offs[:, None] * stride_ot + d_offs[None, :] * stride_od,
        acc.to(tl.float16 if Out_ptr.dtype.element_ty == tl.float16 else tl.bfloat16
               if Out_ptr.dtype.element_ty == tl.bfloat16 else tl.float32),
        mask=mask_qm[:, None] & mask_d[None, :],
    )


# ─── Python 封装 ─────────────────────────────────────────────────────────────

def flash_attn_triton(
    q: torch.Tensor,    # (B, Hq, T_q,  D)
    k: torch.Tensor,    # (B, Hk, T_kv, D)
    v: torch.Tensor,    # (B, Hk, T_kv, D)
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Triton Flash Attention（支持 GQA / causal）。

    prefill:  q.shape[2] == k.shape[2]（== T）
    decode:   q.shape[2] == 1，此时应改用 gemv.py 的 decode_attn

    Returns:
        out: (B, Hq, T_q, D)，dtype 与输入一致
    """
    B, Hq, T_q, D = q.shape
    _, Hk, T_kv, _ = k.shape
    assert Hq % Hk == 0, "GQA 要求 Hq 整除 Hk"
    kv_group_size = Hq // Hk

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # tile 大小：D 对齐到 2 的幂次
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_M = 64    # query  tile（可按 GPU SRAM 调整）
    BLOCK_N = 64    # kv     tile

    out = torch.empty_like(q)

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
    )
    return out
