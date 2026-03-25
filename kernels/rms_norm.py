"""
kernels/rms_norm.py
───────────────────
Triton 实现的 RMSNorm，对标 MiniMind 模型中所有 RMSNorm 调用：
  - 主干 hidden_states 的 input_layernorm / post_attention_layernorm / final norm
  - Attention 内部的 q_norm / k_norm（per-head，dim = head_dim）

设计要点：
  1. 计算阶段在 float32 进行（与 Qwen3 改进后的 PyTorch 版保持一致）
  2. 输出 cast 回输入的原始 dtype（bf16 / fp16 / fp32 均支持）
  3. BLOCK_SIZE 在 kernel 调用时按 dim 取下一个 2 的幂次，充分利用向量化
"""

import torch
import triton
import triton.language as tl


# ─── Triton kernel ────────────────────────────────────────────────────────────

@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr,          # 输入指针，  shape: (M, N)，行优先
    W_ptr,          # weight 指针，shape: (N,)
    Y_ptr,          # 输出指针，  shape: (M, N)
    M,              # 行数（batch * seq_len，或 batch * seq * heads）
    N,              # 列数（dim / head_dim）
    eps,            # rms_norm_eps
    stride_xm,      # X 行 stride（== N，若内存连续）
    stride_ym,      # Y 行 stride
    BLOCK_N: tl.constexpr,  # tile 大小，编译时确定
):
    # 每个 program 处理一行
    row_id = tl.program_id(0)
    X_row = X_ptr + row_id * stride_xm
    Y_row = Y_ptr + row_id * stride_ym

    # 列偏移，超出 N 的用 0 填充（mask 掉）
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 1. 读入一行，转 fp32
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)

    # 2. 计算 RMS（float32 累加，避免 bf16 精度损失）
    variance = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)

    # 3. 归一化 + scale
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    y = (x * rrms) * w

    # 4. 写回，保持输入 dtype（由 triton 自动 cast）
    tl.store(Y_row + cols, y.to(x.dtype), mask=mask)


# ─── Python 封装 ─────────────────────────────────────────────────────────────

def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    与 MiniMind RMSNorm.forward 等价的 Triton 实现。

    Args:
        x:      (..., dim)，任意前缀维度，最后一维是 norm 维
        weight: (dim,)
        eps:    rms_norm_eps

    Returns:
        y: 与 x 相同 shape 和 dtype
    """
    orig_shape = x.shape
    dim = orig_shape[-1]
    # 将所有前缀维度折叠成行
    x_2d = x.contiguous().view(-1, dim)
    M, N = x_2d.shape

    y_2d = torch.empty_like(x_2d)

    # BLOCK_N 取 N 的下一个 2 的幂次，最大 64K（Triton shared memory 限制）
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_N = min(BLOCK_N, 65536)

    grid = (M,)
    _rms_norm_fwd_kernel[grid](
        x_2d, weight, y_2d,
        M, N, eps,
        x_2d.stride(0), y_2d.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return y_2d.view(orig_shape)


# ─── nn.Module 替换接口 ───────────────────────────────────────────────────────

class TritonRMSNorm(torch.nn.Module):
    """
    可直接替换 model_minimind.py 中的 RMSNorm 模块。
    参数名 (weight, eps) 与原版保持一致，权重加载无缝兼容。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_triton(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"
