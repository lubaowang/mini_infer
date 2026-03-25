"""
kernels/gemv.py
───────────────
Triton GEMV（矩阵向量乘）：专门优化 decode 阶段的单 token 推理。

背景：
  decode 阶段 batch=1, seq_len=1，线性层变成 [1, hidden] × [hidden, out]
  即 GEMV（General Matrix-Vector Multiply）。
  PyTorch cublas 默认走 GEMM 路径，对 GEMV 并不是最优。
  Triton GEMV 使用 warp-level reduction，每个 warp 处理一行输出。

覆盖范围：
  - q/k/v/o_proj 的前向（Attention 投影）
  - gate_proj / up_proj / down_proj（MLP 投影）

不适用场景：
  - prefill（seq_len > 1）→ 走 torch.matmul 或 flash_attn
  - 批量推理（batch > 1）→ 走 torch.matmul
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    A_ptr,          # weight matrix,  (N_out, N_in)，row-major
    X_ptr,          # input vector,   (N_in,)
    Out_ptr,        # output vector,  (N_out,)
    N_in, N_out,
    BLOCK_K: tl.constexpr,  # 每个 program 处理的 N_in tile
):
    """
    每个 program 负责输出的一行（即 A 的一行与 X 的点积）。
    多个 program 组成完整的 N_out 输出。
    """
    row = tl.program_id(0)
    if row >= N_out:
        return

    # 分段累加，处理任意长度的 N_in
    acc = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, N_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < N_in

        a = tl.load(A_ptr + row * N_in + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + k_offs,              mask=mask_k, other=0.0).to(tl.float32)
        acc += tl.sum(a * x, axis=0)

    tl.store(Out_ptr + row, acc.to(tl.float32))


def gemv_triton(
    weight: torch.Tensor,  # (N_out, N_in)，Linear 的 weight（已转置！）
    x: torch.Tensor,       # (N_in,) 或 (1, N_in)
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    decode 阶段 Linear 前向：y = x @ weight.T （+ bias）

    等价于 F.linear(x, weight, bias)，但针对 batch=1 的 GEMV 做了优化。

    Args:
        weight: (N_out, N_in)，即 nn.Linear.weight 的形状
        x:      (N_in,) 或 (1, N_in)
        bias:   (N_out,) 或 None

    Returns:
        (N_out,) 或 (1, N_out) 与 x 的前缀维度一致
    """
    squeeze = False
    if x.dim() == 2:
        assert x.shape[0] == 1
        x = x.squeeze(0)
        squeeze = True

    N_in  = weight.shape[1]
    N_out = weight.shape[0]
    assert x.shape[0] == N_in

    out = torch.empty(N_out, device=x.device, dtype=torch.float32)

    BLOCK_K = min(triton.next_power_of_2(N_in), 1024)
    grid = (N_out,)
    _gemv_kernel[grid](
        weight, x, out,
        N_in, N_out,
        BLOCK_K=BLOCK_K,
    )

    # cast 回原始 dtype
    out = out.to(x.dtype)

    if bias is not None:
        out = out + bias

    return out.unsqueeze(0) if squeeze else out


# ─── decode 阶段 Attention（单 token，KV-cache 已填充） ────────────────────

@triton.jit
def _decode_attn_kernel(
    Q_ptr,          # (Hq, D)
    K_ptr,          # (Hk, T_kv, D)  ← KV cache
    V_ptr,          # (Hk, T_kv, D)
    Out_ptr,        # (Hq, D)
    Hq, Hk, T_kv, D,
    scale,
    kv_group_size,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    decode 阶段 attention：q 只有 1 个 token，对所有历史 KV 做 softmax attention。
    每个 program 处理一个 q head。
    """
    hq = tl.program_id(0)
    hk = hq // kv_group_size

    d_offs = tl.arange(0, BLOCK_D)
    mask_d = d_offs < D

    # 读 q: (D,)
    q = tl.load(Q_ptr + hq * D + d_offs, mask=mask_d, other=0.0).to(tl.float32) * scale

    # online softmax across T_kv
    m_i = float("-inf")
    l_i = 0.0
    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t_start in range(0, T_kv, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        mask_t = t_offs < T_kv

        # 读 K tile: (BLOCK_T, D)
        k_base = K_ptr + hk * T_kv * D
        k = tl.load(
            k_base + t_offs[:, None] * D + d_offs[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # score: (BLOCK_T,) = q @ k^T
        scores = tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(mask_t, scores, float("-inf"))

        # online softmax
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        exp_s = tl.exp(scores - m_new)
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_s, axis=0)

        # 读 V tile: (BLOCK_T, D)
        v_base = V_ptr + hk * T_kv * D
        v = tl.load(
            v_base + t_offs[:, None] * D + d_offs[None, :],
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # rescale acc
        acc = acc * (tl.exp(m_i - m_new) * l_i / l_new) + tl.sum(exp_s[:, None] * v, axis=0) / l_new
        m_i = m_new
        l_i = l_new

    tl.store(Out_ptr + hq * D + d_offs, acc.to(tl.float32), mask=mask_d)


def decode_attn_triton(
    q: torch.Tensor,    # (B=1, Hq, 1, D)  — decode 单 token
    k: torch.Tensor,    # (B=1, Hk, T_kv, D) — KV cache（含当前 token）
    v: torch.Tensor,    # (B=1, Hk, T_kv, D)
    scale: float | None = None,
) -> torch.Tensor:
    """
    decode 阶段专用 attention，B=1, T_q=1。
    Returns: (1, Hq, 1, D)
    """
    assert q.shape[0] == 1 and q.shape[2] == 1
    Hq, Hk = q.shape[1], k.shape[1]
    T_kv, D = k.shape[2], k.shape[3]
    kv_group_size = Hq // Hk
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    q_2d   = q.squeeze(0).squeeze(1)           # (Hq, D)
    k_3d   = k.squeeze(0).contiguous()         # (Hk, T_kv, D)
    v_3d   = v.squeeze(0).contiguous()         # (Hk, T_kv, D)
    out_2d = torch.empty(Hq, D, device=q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_T = min(triton.next_power_of_2(T_kv), 128)

    _decode_attn_kernel[(Hq,)](
        q_2d, k_3d, v_3d, out_2d,
        Hq, Hk, T_kv, D,
        scale, kv_group_size,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return out_2d.to(q.dtype).unsqueeze(0).unsqueeze(2)   # (1, Hq, 1, D)
