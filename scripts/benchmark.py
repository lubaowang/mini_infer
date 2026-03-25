"""
scripts/benchmark.py
────────────────────
算子级 + 端到端性能基准测试。
对比：Triton 算子 vs PyTorch（SDPA / cublas）的延迟和吞吐。

运行：
    python scripts/benchmark.py                         # 默认配置
    python scripts/benchmark.py --hidden_size 512 --seq_len 2048
    python scripts/benchmark.py --mode e2e             # 端到端 tokens/s
"""

import argparse
import math
import time
from typing import Callable

import torch

try:
    import triton  # noqa
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─── 计时工具 ──────────────────────────────────────────────────────────────────

def cuda_timer(fn: Callable, warmup: int = 5, repeat: int = 50) -> float:
    """返回 fn() 的平均延迟（ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat   # ms


def print_table(rows: list[tuple], headers: list[str]):
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, floatfmt=".3f", tablefmt="github"))
    except ImportError:
        # fallback 简单格式
        col_w = max(len(h) for h in headers) + 4
        print("  ".join(h.ljust(col_w) for h in headers))
        print("-" * (col_w * len(headers)))
        for row in rows:
            print("  ".join(str(v).ljust(col_w) for v in row))


# ─── 单算子 benchmark ─────────────────────────────────────────────────────────

def bench_rms_norm(hidden_size: int, seq_len: int, batch: int = 1):
    if not HAS_TRITON:
        print("[skip] RMSNorm: Triton not available"); return
    from kernels.rms_norm import rms_norm_triton

    x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    w = torch.ones(hidden_size, device="cuda", dtype=torch.float16)

    def ref():
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (w.float() * x.float() * torch.rsqrt(v + 1e-5)).half()

    def triton_fn():
        return rms_norm_triton(x, w, 1e-5)

    t_ref    = cuda_timer(ref)
    t_triton = cuda_timer(triton_fn)
    return [("RMSNorm", hidden_size, seq_len, f"{t_ref:.3f}", f"{t_triton:.3f}", f"{t_ref/t_triton:.2f}x")]


def bench_flash_attn(hidden_size: int, seq_len: int, num_heads: int = 8, num_kv_heads: int = 2, batch: int = 1):
    if not HAS_TRITON:
        print("[skip] FlashAttn: Triton not available"); return
    from kernels.flash_attn_triton import flash_attn_triton
    import torch.nn.functional as F

    head_dim = hidden_size // num_heads
    q = torch.randn(batch, num_heads,    seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(head_dim)
    n_rep = num_heads // num_kv_heads

    def ref():
        k_exp = k.repeat_interleave(n_rep, dim=1)
        v_exp = v.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True, scale=scale)

    def triton_fn():
        return flash_attn_triton(q, k, v, causal=True, scale=scale)

    t_ref    = cuda_timer(ref)
    t_triton = cuda_timer(triton_fn)
    return [("FlashAttn(prefill)", f"{seq_len}T", f"H={num_heads}/Hk={num_kv_heads}",
             f"{t_ref:.3f}", f"{t_triton:.3f}", f"{t_ref/t_triton:.2f}x")]


def bench_silu_gate(hidden_size: int, intermediate_size: int, batch: int = 1, seq_len: int = 512):
    if not HAS_TRITON:
        print("[skip] SiLU-Gate: Triton not available"); return
    from kernels.silu_gate import silu_gate_triton
    import torch.nn.functional as F

    M = batch * seq_len
    gate = torch.randn(M, intermediate_size, device="cuda", dtype=torch.float16)
    up   = torch.randn(M, intermediate_size, device="cuda", dtype=torch.float16)

    def ref():
        return F.silu(gate) * up

    def triton_fn():
        return silu_gate_triton(gate, up)

    t_ref    = cuda_timer(ref)
    t_triton = cuda_timer(triton_fn)
    return [("SiLU-Gate(SwiGLU)", intermediate_size, seq_len,
             f"{t_ref:.3f}", f"{t_triton:.3f}", f"{t_ref/t_triton:.2f}x")]


def bench_gemv(hidden_size: int, intermediate_size: int):
    if not HAS_TRITON:
        print("[skip] GEMV: Triton not available"); return
    from kernels.gemv import gemv_triton
    import torch.nn.functional as F

    weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16)
    x      = torch.randn(1, hidden_size,                 device="cuda", dtype=torch.float16)

    def ref():
        return F.linear(x, weight)

    def triton_fn():
        return gemv_triton(weight, x)

    t_ref    = cuda_timer(ref)
    t_triton = cuda_timer(triton_fn)
    return [("GEMV(decode proj)", hidden_size, intermediate_size,
             f"{t_ref:.3f}", f"{t_triton:.3f}", f"{t_ref/t_triton:.2f}x")]


# ─── 端到端 tokens/s benchmark ────────────────────────────────────────────────

def bench_e2e(args):
    """比较 Triton 引擎 vs 原生 PyTorch 在 tokens/s 上的差距。"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from engine import MiniMindInferenceEngine, StaticKVCache, patch_model

    cfg = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    device = "cuda"
    dtype  = torch.float16
    prompt_len    = 32
    new_tokens    = 100
    input_ids     = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    results = []
    for use_triton in ([False, True] if HAS_TRITON else [False]):
        tag = "Triton" if use_triton else "PyTorch"
        model = MiniMindForCausalLM(cfg).to(device=device, dtype=dtype).eval()
        kv_cache = StaticKVCache.new(
            cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
            max_seq=2048, device=torch.device(device), dtype=dtype,
        )
        engine = MiniMindInferenceEngine(model, kv_cache, use_triton=use_triton)

        # warmup
        engine.generate(input_ids, max_new_tokens=20)

        st = time.perf_counter()
        for _ in range(5):
            engine.generate(input_ids, max_new_tokens=new_tokens)
        elapsed = (time.perf_counter() - st) / 5

        tps = new_tokens / elapsed
        results.append((tag, args.hidden_size, f"{tps:.1f}", f"{elapsed*1000:.1f}"))

    print("\n── 端到端 tokens/s ──")
    print_table(results, ["Backend", "hidden_size", "tokens/s", "latency(ms/req)"])


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size",       default=512,  type=int)
    parser.add_argument("--num_hidden_layers", default=8,    type=int)
    parser.add_argument("--seq_len",           default=512,  type=int)
    parser.add_argument("--mode", default="ops", choices=["ops", "e2e", "all"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ 需要 CUDA GPU，当前无可用 GPU"); return

    intermediate = int(args.hidden_size * 8 / 3)
    intermediate = 64 * ((intermediate + 63) // 64)

    if args.mode in ("ops", "all"):
        all_rows = []

        r = bench_rms_norm(args.hidden_size, args.seq_len)
        if r: all_rows.extend(r)

        r = bench_flash_attn(args.hidden_size, args.seq_len)
        if r: all_rows.extend(r)

        r = bench_silu_gate(args.hidden_size, intermediate, seq_len=args.seq_len)
        if r: all_rows.extend(r)

        r = bench_gemv(args.hidden_size, intermediate)
        if r: all_rows.extend(r)

        if all_rows:
            print("\n── 算子 Latency 对比（单位 ms，越小越好） ──")
            print_table(
                all_rows,
                ["Op", "size1", "size2/seq", "PyTorch(ms)", "Triton(ms)", "Speedup"]
            )

    if args.mode in ("e2e", "all"):
        bench_e2e(args)


if __name__ == "__main__":
    main()
