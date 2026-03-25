"""
kernels/silu_gate.py
────────────────────
Triton 实现的 Fused SiLU-Gate（SwiGLU MLP 的中间步骤）。

对标 MiniMind FeedForward.forward 中的：
    self.act_fn(self.gate_proj(x)) * self.up_proj(x)

即：element-wise  silu(gate) * up

设计：
  - 将 gate 和 up 的 linear 输出 concat 后一次性送入 kernel，避免两次 store + load
  - 也支持分开传入（gate_out, up_out 各自 (M, intermediate_size)）
  - 原地或新建输出均支持
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_gate_fwd_kernel(
    Gate_ptr,   # gate_proj 输出，(M, N)，等待 silu
    Up_ptr,     # up_proj   输出，(M, N)
    Out_ptr,    # fused 输出，   (M, N)
    M, N,
    stride_m,   # 行 stride（所有 tensor 行 stride 相同，要求等 N 且连续）
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    base = row * stride_m

    gate = tl.load(Gate_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(Up_ptr   + base + cols, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu_gate = gate * tl.sigmoid(gate)

    out = silu_gate * up

    tl.store(Out_ptr + base + cols, out.to(gate.dtype), mask=mask)


def silu_gate_triton(
    gate_out: torch.Tensor,   # (M, intermediate_size)，gate_proj 的输出
    up_out:   torch.Tensor,   # (M, intermediate_size)，up_proj   的输出
) -> torch.Tensor:
    """
    等价于：ACT2FN['silu'](gate_out) * up_out

    输入需连续（调用前可 .contiguous()）。
    """
    assert gate_out.shape == up_out.shape
    M, N = gate_out.shape
    out  = torch.empty_like(gate_out)
    BLOCK_N = min(triton.next_power_of_2(N), 4096)

    _silu_gate_fwd_kernel[(M,)](
        gate_out, up_out, out,
        M, N, gate_out.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return out
