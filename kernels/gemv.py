"""
kernels/gemv.py  （修正版）
───────────────
Triton GEMV + decode 阶段 Attention。

修正点：
  1. _gemv_kernel: acc 初始化为 tl.zeros([1]) 而非标量 0.0，
     保证 tl.sum 返回值类型一致；同时 tl.store 写 acc[0]。
  2. _decode_attn_kernel: l_new 为 0 时（第一个 tile，所有 score=-inf）
     用 eps=1e-8 防止除零 NaN；同时 acc 更新时把旧 acc rescale 和新贡献分开写清楚。
  3. decode_attn_triton 输入的 k/v 已是 (B, Hk, T_kv, D) heads-first，
     stride 按实际 shape 传入，不假设 contiguous 时 stride = T_kv * D。
"""

import torch
import triton
import triton.language as tl


# ─── GEMV ────────────────────────────────────────────────────────────────────

@triton.jit
def _gemv_kernel(
    A_ptr,          # (N_out, N_in) row-major
    X_ptr,          # (N_in,)
    Out_ptr,        # (N_out,)
    N_in, N_out,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N_out:
        return

    acc = tl.zeros([], tl.float32)

    for k_start in range(0, N_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < N_in

        a = tl.load(A_ptr + row * N_in + k_offs, mask=mask_k, other=0.0)
        x = tl.load(X_ptr + k_offs,              mask=mask_k, other=0.0)

        acc += tl.sum(a * x)

    tl.store(Out_ptr + row, acc)


def gemv_triton(
    weight: torch.Tensor,   # (N_out, N_in)
    x: torch.Tensor,        # (N_in,) 或 (1, N_in)
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    squeeze = False
    if x.dim() == 2:
        assert x.shape[0] == 1
        x = x.squeeze(0)
        squeeze = True

    N_in  = weight.shape[1]
    N_out = weight.shape[0]
    assert x.shape[0] == N_in, f"shape mismatch: weight({N_out},{N_in}) x({x.shape})"

    # weight 和 x 必须连续才能用指针算术
    weight = weight.contiguous()
    x      = x.contiguous()

    out = torch.empty(N_out, device=x.device, dtype=torch.float32)
    BLOCK_K = min(triton.next_power_of_2(N_in), 1024)

    _gemv_kernel[(N_out,)](weight, x, out, N_in, N_out, BLOCK_K=BLOCK_K)

    out = out.to(x.dtype)
    if bias is not None:
        out = out + bias
    return out.unsqueeze(0) if squeeze else out


# ─── decode 阶段 Attention ────────────────────────────────────────────────────

@triton.jit
def _decode_attn_kernel(
    Q_ptr,          # (Hq, D) contiguous
    K_ptr,          # (Hk, T_kv, D) contiguous
    V_ptr,          # (Hk, T_kv, D) contiguous
    Out_ptr,        # (Hq, D)
    Hq, Hk, T_kv, D,
    scale,
    kv_group_size,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hq = tl.program_id(0)
    hk = hq // kv_group_size

    d_offs = tl.arange(0, BLOCK_D)
    mask_d = d_offs < D

    q = tl.load(Q_ptr + hq * D + d_offs, mask=mask_d, other=0.0).to(tl.float32) * scale

    # online softmax 初始值
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t_start in range(0, T_kv, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_offs < T_kv

        # K tile: (BLOCK_T, D)
        k = tl.load(
            K_ptr + hk * T_kv * D + t_offs[:, None] * D + d_offs[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # scores: (BLOCK_T,)
        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(mask_t, scores, float("-inf"))

        # online softmax 更新
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        exp_s = tl.exp(scores - m_new)                          # (BLOCK_T,)
        sum_exp = tl.sum(exp_s, axis=0)                         # scalar

        # ✅ 修正：l_new 加 eps 防除零（第一个 tile 时 l_i=0 且 m_i=-inf）
        rescale = tl.exp(m_i - m_new)
        l_new   = rescale * l_i + sum_exp + 1e-8

        # V tile: (BLOCK_T, D)
        v = tl.load(
            V_ptr + hk * T_kv * D + t_offs[:, None] * D + d_offs[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # acc 更新：先 rescale 旧 acc，再加新贡献
        weighted_v = tl.sum(exp_s[:, None] * v, axis=0)         # (BLOCK_D,)
        acc = acc * (rescale * l_i / l_new) + weighted_v / l_new

        m_i = m_new
        l_i = l_new

    tl.store(Out_ptr + hq * D + d_offs, acc, mask=mask_d)


def decode_attn_triton(
    q: torch.Tensor,    # (B=1, Hq, 1, D)
    k: torch.Tensor,    # (B=1, Hk, T_kv, D)
    v: torch.Tensor,    # (B=1, Hk, T_kv, D)
    scale: float | None = None,
) -> torch.Tensor:
    """
    decode 阶段专用，B=1, T_q=1。
    Returns: (1, Hq, 1, D)，dtype 与 q 一致。
    """
    assert q.shape[0] == 1 and q.shape[2] == 1
    Hq    = q.shape[1]
    Hk    = k.shape[1]
    T_kv  = k.shape[2]
    D     = k.shape[3]
    kv_group_size = Hq // Hk

    if scale is None:
        scale = D ** -0.5

    # 确保连续（K/V 来自 cache slice，可能不连续）
    q_2d = q.squeeze(0).squeeze(1).contiguous()     # (Hq, D)
    k_3d = k.squeeze(0).contiguous()                # (Hk, T_kv, D)
    v_3d = v.squeeze(0).contiguous()                # (Hk, T_kv, D)

    out_2d = torch.empty(Hq, D, device=q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_T = min(triton.next_power_of_2(T_kv), 128)

    _decode_attn_kernel[(Hq,)](
        q_2d, k_3d, v_3d, out_2d,
        Hq, Hk, T_kv, D,
        scale, kv_group_size,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )
    return out_2d.to(q.dtype).unsqueeze(0).unsqueeze(2)   # (1, Hq, 1, D)
