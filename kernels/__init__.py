"""
kernels/__init__.py
───────────────────
统一暴露所有 Triton 算子，并提供 TRITON_AVAILABLE 标志。
推理引擎通过此模块导入算子，运行时若 Triton 不可用则自动回退到 PyTorch。
"""

import warnings

TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    warnings.warn(
        "[minimind_infer] Triton 未安装或当前设备不支持（CPU 模式）。"
        "所有算子将自动退回到 PyTorch 实现。",
        RuntimeWarning,
        stacklevel=2,
    )

if TRITON_AVAILABLE:
    from .rms_norm         import rms_norm_triton, TritonRMSNorm
    from .rope_emb         import apply_rotary_pos_emb_triton
    from .flash_attn_triton import flash_attn_triton
    from .silu_gate        import silu_gate_triton
    from .gemv             import gemv_triton, decode_attn_triton
else:
    # ── PyTorch fallback stubs ──────────────────────────────────────────────
    import torch
    import torch.nn.functional as F
    import math

    def rms_norm_triton(x, weight, eps=1e-5):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(input_dtype)

    class TritonRMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return rms_norm_triton(x, self.weight, self.eps)

    def apply_rotary_pos_emb_triton(q, k, cos, sin):
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def flash_attn_triton(q, k, v, causal=True, scale=None):
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    def silu_gate_triton(gate_out, up_out):
        return F.silu(gate_out) * up_out

    def gemv_triton(weight, x, bias=None):
        return F.linear(x, weight, bias)

    def decode_attn_triton(q, k, v, scale=None):
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)


__all__ = [
    "TRITON_AVAILABLE",
    "rms_norm_triton",
    "TritonRMSNorm",
    "apply_rotary_pos_emb_triton",
    "flash_attn_triton",
    "silu_gate_triton",
    "gemv_triton",
    "decode_attn_triton",
]
