"""
tests/test_kernels.py
─────────────────────
验证每个 Triton 算子的数值正确性：
  对比 Triton 输出与 PyTorch 参考实现，在 fp16/bf16/fp32 下分别断言 atol/rtol。

运行：
    pytest tests/test_kernels.py -v
    pytest tests/test_kernels.py -v -k "rms"   # 只跑 rms_norm 相关
"""

import math
import pytest
import torch
import torch.nn.functional as F

# 跳过没有 GPU 的环境
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require CUDA GPU"
)

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

skip_no_triton = pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")


# ─── RMSNorm ─────────────────────────────────────────────────────────────────

@skip_no_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 64, 512), (1, 1, 512), (2, 16, 64)])
def test_rms_norm(shape, dtype):
    from kernels.rms_norm import rms_norm_triton
    dim = shape[-1]
    x   = torch.randn(*shape, device="cuda", dtype=dtype)
    w   = torch.ones(dim, device="cuda", dtype=dtype)
    eps = 1e-5

    # PyTorch 参考
    x_fp32    = x.float()
    variance  = x_fp32.pow(2).mean(-1, keepdim=True)
    ref       = (w.float() * x_fp32 * torch.rsqrt(variance + eps)).to(dtype)

    out = rms_norm_triton(x, w, eps)

    atol = 1e-2 if dtype == torch.float16 else (3e-3 if dtype == torch.bfloat16 else 1e-5)
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=atol), \
        f"RMSNorm mismatch: max_diff={( out.float()-ref.float()).abs().max().item():.2e}"


# ─── SiLU-Gate（SwiGLU） ──────────────────────────────────────────────────────

@skip_no_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M,N", [(8, 1024), (1, 2048), (32, 512)])
def test_silu_gate(M, N, dtype):
    from kernels.silu_gate import silu_gate_triton
    gate = torch.randn(M, N, device="cuda", dtype=dtype)
    up   = torch.randn(M, N, device="cuda", dtype=dtype)

    ref = F.silu(gate.float()) * up.float()
    out = silu_gate_triton(gate.contiguous(), up.contiguous())

    atol = 1e-2 if dtype == torch.float16 else 1e-5
    assert torch.allclose(out.float(), ref.float(), atol=atol), \
        f"SiLU-Gate mismatch: max_diff={(out.float()-ref.float()).abs().max().item():.2e}"


# ─── RoPE ─────────────────────────────────────────────────────────────────────

@skip_no_triton
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("B,Hq,Hk,T,D", [
    (1, 8, 2, 16, 64),   # GQA: Hq=8, Hk=2
    (2, 4, 4, 32, 64),   # MHA: Hq==Hk
    (1, 8, 2,  1, 64),   # decode: T=1
])
def test_rope(B, Hq, Hk, T, D, dtype):
    from kernels.rope_emb import apply_rotary_pos_emb_triton

    q = torch.randn(B, Hq, T, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Hk, T, D, device="cuda", dtype=dtype)

    # 构造 cos/sin: (B, T, D)
    pos = torch.arange(T, device="cuda").unsqueeze(0).expand(B, -1)   # (B, T)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    freqs = torch.outer(pos.reshape(-1), inv_freq).view(B, T, -1)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)

    # PyTorch 参考
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q_ref = (q.float() * c.float()) + (rotate_half(q.float()) * s.float())
    k_ref = (k.float() * c.float()) + (rotate_half(k.float()) * s.float())

    q_out, k_out = apply_rotary_pos_emb_triton(q, k, cos, sin)

    atol = 1e-2
    assert torch.allclose(q_out.float(), q_ref.to(q_out.dtype).float(), atol=atol), \
        f"RoPE q mismatch: max={(q_out.float()-q_ref.float()).abs().max().item():.2e}"
    assert torch.allclose(k_out.float(), k_ref.to(k_out.dtype).float(), atol=atol), \
        f"RoPE k mismatch: max={(k_out.float()-k_ref.float()).abs().max().item():.2e}"


# ─── Flash Attention ─────────────────────────────────────────────────────────

@skip_no_triton
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("B,Hq,Hk,T,D", [
    (1, 8, 2,  64, 64),
    (2, 4, 4, 128, 64),
    (1, 4, 2, 256, 32),
])
def test_flash_attn_causal(B, Hq, Hk, T, D, dtype):
    from kernels.flash_attn_triton import flash_attn_triton
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, Hq, T, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Hk, T, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Hk, T, D, device="cuda", dtype=dtype)

    # GQA expand for reference
    n_rep = Hq // Hk
    k_exp = k.repeat_interleave(n_rep, dim=1)
    v_exp = v.repeat_interleave(n_rep, dim=1)

    # PyTorch SDPA 参考
    ref = F.scaled_dot_product_attention(q.float(), k_exp.float(), v_exp.float(),
                                          is_causal=True, scale=scale).to(dtype)

    out = flash_attn_triton(q, k, v, causal=True, scale=scale)

    # Flash Attention 允许相对较大的数值误差（tiling 累积）
    atol = 0.05
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=atol), \
        f"FlashAttn mismatch: max={(out.float()-ref.float()).abs().max().item():.3f}"


# ─── GEMV（decode 阶段 Linear） ────────────────────────────────────────────────

@skip_no_triton
@pytest.mark.parametrize("N_in,N_out", [(512, 512), (512, 2048), (2048, 512)])
def test_gemv(N_in, N_out):
    from kernels.gemv import gemv_triton
    weight = torch.randn(N_out, N_in, device="cuda", dtype=torch.float16)
    x      = torch.randn(1, N_in,     device="cuda", dtype=torch.float16)

    ref = F.linear(x, weight)
    out = gemv_triton(weight, x)

    assert torch.allclose(out.float(), ref.float(), atol=1e-2), \
        f"GEMV mismatch: max={(out.float()-ref.float()).abs().max().item():.2e}"


# ─── decode_attn ─────────────────────────────────────────────────────────────

@skip_no_triton
@pytest.mark.parametrize("Hq,Hk,T_kv,D", [
    (8, 2, 64, 64),
    (4, 4, 128, 32),
])
def test_decode_attn(Hq, Hk, T_kv, D):
    from kernels.gemv import decode_attn_triton
    B = 1
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, Hq,  1,    D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hk, T_kv,  D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Hk, T_kv,  D, device="cuda", dtype=torch.float16)

    n_rep = Hq // Hk
    k_exp = k.repeat_interleave(n_rep, dim=1)
    v_exp = v.repeat_interleave(n_rep, dim=1)

    ref = F.scaled_dot_product_attention(
        q.float(), k_exp.float(), v_exp.float(), is_causal=False, scale=scale
    ).to(torch.float16)

    out = decode_attn_triton(q, k, v, scale=scale)

    assert torch.allclose(out.float(), ref.float(), atol=0.05), \
        f"DecodeAttn mismatch: max={(out.float()-ref.float()).abs().max().item():.3f}"
