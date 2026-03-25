"""
kernels/__init__.py
───────────────────
统一暴露所有 Triton 算子，并提供 TRITON_AVAILABLE 标志。

修正：增加 CUDA GPU 可用性检查（Triton 安装了但没 GPU 时不可用）。
"""

import warnings
import torch

TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    # ✅ Triton 需要 CUDA GPU 才能工作
    if torch.cuda.is_available():
        TRITON_AVAILABLE = True
    else:
        warnings.warn(
            "[minimind_infer] Triton 已安装，但当前无 CUDA GPU，"
            "自动回退到 PyTorch 实现。",
            RuntimeWarning, stacklevel=2,
        )
except ImportError:
    warnings.warn(
        "[minimind_infer] Triton 未安装，自动回退到 PyTorch 实现。",
        RuntimeWarning, stacklevel=2,
    )

if TRITON_AVAILABLE:
    from .rms_norm          import rms_norm_triton, TritonRMSNorm
    from .rope_emb          import apply_rotary_pos_emb_triton
    from .flash_attn_triton import flash_attn_triton
    from .silu_gate         import silu_gate_triton
    from .gemv              import gemv_triton, decode_attn_triton
else:
    # ── PyTorch fallback stubs ──────────────────────────────────────────────
    import math
    import torch.nn.functional as F

    def rms_norm_triton(x, weight, eps=1e-5):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight.to(torch.float32) * x).to(input_dtype)

    class TritonRMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.eps    = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return rms_norm_triton(x, self.weight, self.eps)

    def apply_rotary_pos_emb_triton(q, k, cos, sin):
        def rotate_half(t):
            t1, t2 = t[..., :t.shape[-1]//2], t[..., t.shape[-1]//2:]
            return torch.cat((-t2, t1), dim=-1)
        c = cos.unsqueeze(1)   # (B,1,T,D)
        s = sin.unsqueeze(1)
        return (q * c + rotate_half(q) * s), (k * c + rotate_half(k) * s)

    def flash_attn_triton(q, k, v, causal=True, scale=None):
        if scale is None:
            scale = q.shape[-1] ** -0.5
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    def silu_gate_triton(gate_out, up_out):
        return F.silu(gate_out) * up_out

    def gemv_triton(weight, x, bias=None):
        return F.linear(x, weight, bias)

    def decode_attn_triton(q, k, v, scale=None):
        if scale is None:
            scale = q.shape[-1] ** -0.5
        n_rep = q.shape[1] // k.shape[1]
        k_exp = k.repeat_interleave(n_rep, dim=1)
        v_exp = v.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False, scale=scale)


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
