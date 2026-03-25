"""
engine/inference_engine.py  （最终修正版）
──────────────────────────

Bug 修复总结：
  1. KV Cache 时机：改用 update_and_get() 在 forward 内立即写入并返回完整 KV，
     forward 结束后调用 step(T_new) 推进指针。

  2. RoPE position_ids 错误：decode 时 cur_pos 通过引擎显式追踪并传入
     _model_forward_with_pos，保证每步 position_ids 从正确偏移开始。

  3. _is_decode 传参问题：不通过 forward() kwargs 传递（原始模型不支持此参数），
     改为在 monkey-patch 的闭包内通过一个可变对象（list cell）来共享状态。
     引擎在调用 forward 前设置 _decode_flag[0]，patch 函数从闭包读取。

  4. GEMV 输出 shape 一致性：decode 时 o_proj GEMV 输入 (B, hidden)，
     输出 (B, hidden)，unsqueeze(1) 后与 non-decode 路径的 (B,1,hidden) 对齐。

  5. PyTorch fallback（CPU）完整性：不 patch 时 _model_forward_with_pos
     直接调用原始 block.self_attn.forward，is_decode 信息通过 past_key_value 接口
     不传入（原模型不支持 cache-less decode），fallback 路径仅用于测试正确性，
     生产环境走 GPU + Triton。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List

from kernels import (
    TRITON_AVAILABLE,
    TritonRMSNorm,
    apply_rotary_pos_emb_triton,
    flash_attn_triton,
    silu_gate_triton,
    gemv_triton,
    decode_attn_triton,
)
from engine.kv_cache import StaticKVCache
from engine.sampler import sample


# ─── Triton FeedForward ───────────────────────────────────────────────────────

class TritonFeedForward(nn.Module):
    def __init__(self, ff: nn.Module):
        super().__init__()
        self.gate_proj = ff.gate_proj
        self.up_proj   = ff.up_proj
        self.down_proj = ff.down_proj
        self.dropout   = ff.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2   = x.view(-1, x.shape[-1])
        g    = self.gate_proj(x2)
        u    = self.up_proj(x2)
        if TRITON_AVAILABLE and x2.is_cuda:
            act = silu_gate_triton(g.contiguous(), u.contiguous())
        else:
            act = F.silu(g) * u
        return self.dropout(self.down_proj(act)).view(orig)


# ─── Triton Attention forward（monkey-patch） ─────────────────────────────────

def _make_triton_attn_forward(kv_cache: StaticKVCache, layer_idx: int, decode_flag: list):
    """
    decode_flag: [False]，可变 list cell，引擎在调用前设置 decode_flag[0]。
    闭包读取它，避免通过 forward() 参数传递（原始 Attention.forward 签名不含此参数）。
    """
    def triton_attn_forward(
        self,
        x: torch.Tensor,
        position_embeddings,
        past_key_value=None,
        use_cache: bool = False,
        attention_mask=None,
    ):
        bsz, seq_len, _ = x.shape
        is_decode      = decode_flag[0]          # ✅ 从闭包读取
        use_triton_ops = TRITON_AVAILABLE and x.is_cuda

        # ── 投影 ────────────────────────────────────────────────────────────
        if is_decode and use_triton_ops:
            x_1d = x.squeeze(1)
            xq = gemv_triton(self.q_proj.weight, x_1d).unsqueeze(1)
            xk = gemv_triton(self.k_proj.weight, x_1d).unsqueeze(1)
            xv = gemv_triton(self.v_proj.weight, x_1d).unsqueeze(1)
        else:
            xq = self.q_proj(x)
            xk = self.k_proj(x)
            xv = self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads,    self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # ── Q/K Norm ────────────────────────────────────────────────────────
        if use_triton_ops:
            from kernels import rms_norm_triton
            xq = rms_norm_triton(xq, self.q_norm.weight, self.q_norm.eps)
            xk = rms_norm_triton(xk, self.k_norm.weight, self.k_norm.eps)
        else:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # ── heads-first + RoPE ──────────────────────────────────────────────
        xq = xq.transpose(1, 2).contiguous()
        xk = xk.transpose(1, 2).contiguous()
        xv = xv.transpose(1, 2).contiguous()

        cos, sin = position_embeddings    # (B, T, D)
        if use_triton_ops:
            xq, xk = apply_rotary_pos_emb_triton(xq, xk, cos, sin)
        else:
            def rotate_half(t):
                t1, t2 = t[..., :t.shape[-1]//2], t[..., t.shape[-1]//2:]
                return torch.cat((-t2, t1), dim=-1)
            c = cos.unsqueeze(1); s = sin.unsqueeze(1)
            xq = xq * c + rotate_half(xq) * s
            xk = xk * c + rotate_half(xk) * s

        # ── KV Cache：立即写入，立即取完整切片 ──────────────────────────────
        full_k, full_v = kv_cache.update_and_get(layer_idx, xk, xv)

        # ── Attention ────────────────────────────────────────────────────────
        if is_decode and use_triton_ops:
            output = decode_attn_triton(xq, full_k, full_v)
        elif use_triton_ops and seq_len > 1:
            output = flash_attn_triton(xq, full_k, full_v, causal=True)
        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            fk    = full_k.repeat_interleave(n_rep, dim=1)
            fv    = full_v.repeat_interleave(n_rep, dim=1)
            output = F.scaled_dot_product_attention(
                xq, fk, fv, is_causal=(not is_decode)
            )

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # ── 输出投影 ─────────────────────────────────────────────────────────
        if is_decode and use_triton_ops:
            output = gemv_triton(self.o_proj.weight, output.squeeze(1)).unsqueeze(1)
        else:
            output = self.o_proj(output)

        output = self.resid_dropout(output)
        return output, None

    return triton_attn_forward


# ─── 模型 patch ───────────────────────────────────────────────────────────────

def patch_model(model: nn.Module, kv_cache: StaticKVCache) -> tuple:
    """
    patch 模型，返回 (patched_model, decode_flag)。
    decode_flag: List[bool]，引擎在每次 forward 前设置 [0] 元素。
    """
    mm_model    = model.model
    decode_flag = [False]   # 全局共享的可变 cell

    # 替换所有 RMSNorm
    def replace_rms(module, name, parent):
        from model.model_minimind import RMSNorm
        if isinstance(module, RMSNorm):
            new_norm        = TritonRMSNorm(module.weight.shape[0], module.eps)
            new_norm.weight = module.weight
            setattr(parent, name, new_norm)
        for cname, child in module.named_children():
            replace_rms(child, cname, module)

    replace_rms(mm_model, "model", model)

    import types
    from model.model_minimind import FeedForward
    for layer_idx, block in enumerate(mm_model.layers):
        if isinstance(block.mlp, FeedForward):
            block.mlp = TritonFeedForward(block.mlp)
        block.self_attn.forward = types.MethodType(
            _make_triton_attn_forward(kv_cache, layer_idx, decode_flag),
            block.self_attn,
        )

    return model, decode_flag


# ─── 驱动 MiniMindModel forward（手动逐层，传入正确 position_ids） ─────────────

def _run_model_forward(
    mm_model,
    input_ids: torch.Tensor,
    cur_pos: int,
    decode_flag: list,
    is_decode: bool,
):
    """
    手动驱动 MiniMindModel 的每一层：
      - 用 cur_pos 构造正确的 position_ids → RotaryEmbedding → cos/sin
      - 设置 decode_flag[0] 供 monkey-patched Attention 读取
      - 处理 patched / unpatched 两种情况（unpatched 时 decode_flag 为 None）
    """
    batch_size, seq_length = input_ids.shape
    hidden = mm_model.dropout(mm_model.embed_tokens(input_ids))

    # ✅ 用 cur_pos 计算正确的 position_ids
    position_ids = torch.arange(
        cur_pos, cur_pos + seq_length, device=input_ids.device
    ).unsqueeze(0)
    position_embeddings = mm_model.rotary_emb(hidden, position_ids)

    if decode_flag is not None:
        decode_flag[0] = is_decode   # ✅ 通知 patched Attention

    from model.model_minimind import MOEFeedForward
    for block in mm_model.layers:
        residual    = hidden
        normed      = block.input_layernorm(hidden)
        attn_out, _ = block.self_attn(
            normed,
            position_embeddings,
            past_key_value=None,
            use_cache=False,
            attention_mask=None,
        )
        hidden = attn_out + residual
        hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))

    hidden = mm_model.norm(hidden)

    aux_loss = sum(
        (l.mlp.aux_loss for l in mm_model.layers if isinstance(l.mlp, MOEFeedForward)),
        hidden.new_zeros(1).squeeze(),
    )
    return hidden, aux_loss


# ─── 推理引擎 ─────────────────────────────────────────────────────────────────

class MiniMindInferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        kv_cache: StaticKVCache,
        use_triton: bool = True,
    ):
        self.model      = model.eval()
        self.kv_cache   = kv_cache
        self.use_triton = use_triton and TRITON_AVAILABLE
        self._decode_flag = None   # 未 patch 时为 None

        if self.use_triton:
            self.model, self._decode_flag = patch_model(self.model, self.kv_cache)

    @classmethod
    def from_checkpoint(
        cls,
        config,
        ckpt_path: str,
        device: str = "cuda",
        max_seq: int = 2048,
        dtype: torch.dtype = torch.float16,
        use_triton: bool = True,
    ) -> "MiniMindInferenceEngine":
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from model.model_minimind import MiniMindForCausalLM

        model = MiniMindForCausalLM(config)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        model = model.to(device=device, dtype=dtype).eval()

        kv_cache = StaticKVCache.new(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_batch=1,
            max_seq=max_seq,
            device=torch.device(device),
            dtype=dtype,
        )
        return cls(model, kv_cache, use_triton=use_triton)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.85,
        do_sample: bool = True,
        eos_token_id: int = 2,
        stream_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        self.kv_cache.reset()
        device        = input_ids.device
        generated_ids: List[int] = []
        prompt_len    = input_ids.shape[1]

        # ── Prefill ──────────────────────────────────────────────────────────
        hidden, _ = _run_model_forward(
            self.model.model, input_ids,
            cur_pos=0, decode_flag=self._decode_flag, is_decode=False,
        )
        self.kv_cache.step(prompt_len)

        logits     = self.model.lm_head(hidden[:, -1:, :])
        next_token = sample(logits, temperature=temperature, top_p=top_p, do_sample=do_sample)
        generated_ids.append(next_token.item())
        if stream_callback:
            stream_callback(next_token.item())

        # ── Decode loop ───────────────────────────────────────────────────────
        cur_pos = prompt_len
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            cur_input = next_token.unsqueeze(0).to(device)   # (1, 1)

            hidden, _ = _run_model_forward(
                self.model.model, cur_input,
                cur_pos=cur_pos, decode_flag=self._decode_flag, is_decode=True,
            )
            self.kv_cache.step(1)
            cur_pos += 1

            logits     = self.model.lm_head(hidden[:, -1:, :])
            next_token = sample(logits, temperature=temperature, top_p=top_p, do_sample=do_sample)
            token_id   = next_token.item()
            generated_ids.append(token_id)
            if stream_callback:
                stream_callback(token_id)

        return generated_ids

    def __repr__(self) -> str:
        return (
            f"MiniMindInferenceEngine("
            f"triton={self.use_triton}, "
            f"kv_cache={self.kv_cache})"
        )
