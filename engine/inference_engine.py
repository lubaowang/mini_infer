"""
engine/inference_engine.py
──────────────────────────
MiniMind Triton 推理引擎核心。

工作流程：
  1. 加载 MiniMindForCausalLM 原始模型权重
  2. 用 Triton 算子 patch 模型中的每个子模块（原地替换，不改变权重）
  3. 维护 StaticKVCache，手动驱动 prefill → decode 循环
  4. 通过 Sampler 完成 token 采样，支持流式输出回调

关键设计：
  - patch_model() 遍历所有 layer，替换 RMSNorm / FeedForward，
    并 monkey-patch Attention.forward 为使用 Triton flash/GEMV 的版本
  - prefill() 一次前向，decode() 逐 token 循环
  - 全程使用 StaticKVCache，不依赖 HuggingFace 的 DynamicCache
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
from contextlib import nullcontext

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


# ─── Triton FeedForward 替换 ─────────────────────────────────────────────────

class TritonFeedForward(nn.Module):
    """
    替换 model_minimind.FeedForward，融合 silu_gate 算子。
    权重直接从原始模块迁移，不需要重新加载。
    """
    def __init__(self, original_ff: nn.Module):
        super().__init__()
        self.gate_proj = original_ff.gate_proj
        self.up_proj   = original_ff.up_proj
        self.down_proj = original_ff.down_proj
        # dropout 在推理时 no-op，保留结构即可
        self.dropout   = original_ff.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        # 折叠 batch 和 seq 维度 → (M, hidden)
        x_2d = x.view(-1, x.shape[-1])

        gate_out = self.gate_proj(x_2d)   # (M, intermediate)
        up_out   = self.up_proj(x_2d)     # (M, intermediate)

        if TRITON_AVAILABLE and x_2d.is_cuda:
            act = silu_gate_triton(gate_out.contiguous(), up_out.contiguous())
        else:
            act = F.silu(gate_out) * up_out

        out = self.down_proj(act)
        return self.dropout(out).view(orig_shape)


# ─── Triton Attention forward patch ──────────────────────────────────────────

def _make_triton_attn_forward(kv_cache: StaticKVCache, layer_idx: int):
    """
    工厂函数：返回一个绑定了 layer_idx 的 Attention.forward。
    这里 monkey-patch 原始 Attention 的 forward，
    复用其所有投影层权重（q/k/v/o_proj, q_norm, k_norm），
    仅替换核心计算路径为 Triton 算子。
    """
    def triton_attn_forward(
        self,
        x: torch.Tensor,
        position_embeddings,
        past_key_value=None,     # 兼容原接口，引擎模式下忽略，用 StaticKVCache 替代
        use_cache: bool = False,
        attention_mask=None,
    ):
        bsz, seq_len, _ = x.shape
        is_decode = (seq_len == 1) and (not kv_cache.is_empty)

        # ── 投影 ────────────────────────────────────────────────────────────
        if is_decode and TRITON_AVAILABLE and x.is_cuda:
            # decode 阶段走 GEMV
            x_1d = x.squeeze(1)   # (B, hidden)
            xq = gemv_triton(self.q_proj.weight, x_1d).unsqueeze(1)
            xk = gemv_triton(self.k_proj.weight, x_1d).unsqueeze(1)
            xv = gemv_triton(self.v_proj.weight, x_1d).unsqueeze(1)
        else:
            xq = self.q_proj(x)
            xk = self.k_proj(x)
            xv = self.v_proj(x)

        # reshape → (B, T, H, D)
        xq = xq.view(bsz, seq_len, self.n_local_heads,    self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # ── Q/K Norm（Qwen3 改进） ───────────────────────────────────────────
        if TRITON_AVAILABLE and x.is_cuda:
            from kernels import rms_norm_triton
            xq = rms_norm_triton(xq, self.q_norm.weight, self.q_norm.eps)
            xk = rms_norm_triton(xk, self.k_norm.weight, self.k_norm.eps)
        else:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # ── 转为 (B, H, T, D) 后施加 RoPE ──────────────────────────────────
        xq = xq.transpose(1, 2).contiguous()   # (B, Hq, T, D)
        xk = xk.transpose(1, 2).contiguous()   # (B, Hk, T, D)
        xv = xv.transpose(1, 2).contiguous()   # (B, Hk, T, D)

        cos, sin = position_embeddings
        # cos/sin: (B, T, D)  → Triton rope 接口需要此格式
        if TRITON_AVAILABLE and x.is_cuda:
            xq, xk = apply_rotary_pos_emb_triton(xq, xk, cos, sin)
        else:
            from kernels import apply_rotary_pos_emb_triton as _rope
            xq, xk = _rope(xq, xk, cos, sin)

        # ── StaticKVCache 更新 ───────────────────────────────────────────────
        kv_cache.update(layer_idx, xk, xv)
        full_k, full_v = kv_cache.get(layer_idx)

        # ── Attention 计算 ───────────────────────────────────────────────────
        if is_decode and TRITON_AVAILABLE and x.is_cuda:
            # decode：单 token，走 Triton GEMV attention
            output = decode_attn_triton(xq, full_k, full_v)  # (B, Hq, 1, D)
        elif TRITON_AVAILABLE and x.is_cuda:
            # prefill：走 Triton flash attention
            output = flash_attn_triton(xq, full_k, full_v, causal=True)  # (B, Hq, T, D)
        else:
            # CPU / fallback：PyTorch SDPA
            n_rep = self.n_local_heads // self.n_local_kv_heads
            full_k_rep = full_k.repeat_interleave(n_rep, dim=1)
            full_v_rep = full_v.repeat_interleave(n_rep, dim=1)
            output = F.scaled_dot_product_attention(xq, full_k_rep, full_v_rep, is_causal=(not is_decode))

        # (B, Hq, T, D) → (B, T, hidden)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # ── 输出投影 ─────────────────────────────────────────────────────────
        if is_decode and TRITON_AVAILABLE and x.is_cuda:
            output = gemv_triton(self.o_proj.weight, output.squeeze(1)).unsqueeze(1)
        else:
            output = self.o_proj(output)

        output = self.resid_dropout(output)
        return output, None   # None 替代 past_kv（引擎自管理 cache）

    return triton_attn_forward


# ─── 模型 patch ───────────────────────────────────────────────────────────────

def patch_model(
    model: nn.Module,
    kv_cache: StaticKVCache,
    use_triton: bool = True,
) -> nn.Module:
    """
    原地 patch 模型：
      1. 所有 RMSNorm → TritonRMSNorm（权重迁移）
      2. 所有 FeedForward → TritonFeedForward（权重共享）
      3. 所有 Attention.forward → triton_attn_forward（monkey-patch）

    不修改参数，仅替换前向计算路径。
    """
    if not use_triton:
        return model

    mm_model = model.model   # MiniMindModel

    # ── 替换主干 RMSNorm ──────────────────────────────────────────────────
    def replace_rms(module: nn.Module, name: str, parent: nn.Module):
        from model.model_minimind import RMSNorm  # 原始 RMSNorm
        if isinstance(module, RMSNorm):
            new_norm = TritonRMSNorm(module.weight.shape[0], module.eps)
            new_norm.weight = module.weight   # 共享权重
            setattr(parent, name, new_norm)
        for child_name, child in module.named_children():
            replace_rms(child, child_name, module)

    replace_rms(mm_model, "model", model)

    # ── 替换 FeedForward & patch Attention ────────────────────────────────
    for layer_idx, block in enumerate(mm_model.layers):
        # FeedForward（非 MoE）
        from model.model_minimind import FeedForward
        if isinstance(block.mlp, FeedForward):
            block.mlp = TritonFeedForward(block.mlp)

        # Attention forward monkey-patch
        import types
        attn = block.self_attn
        attn.forward = types.MethodType(
            _make_triton_attn_forward(kv_cache, layer_idx),
            attn,
        )

    return model


# ─── 推理引擎主类 ──────────────────────────────────────────────────────────────

class MiniMindInferenceEngine:
    """
    高性能推理引擎，对外暴露 generate() 接口，与原 eval.py 的用法接近。

    使用示例：
        engine = MiniMindInferenceEngine.from_checkpoint(
            config=MiniMindConfig(hidden_size=512, ...),
            ckpt_path="out/full_sft_512.pth",
            device="cuda",
        )
        output = engine.generate(
            input_ids=torch.tensor([[1, 2, 3]]).cuda(),
            max_new_tokens=200,
            temperature=0.85,
            top_p=0.85,
        )
    """

    def __init__(
        self,
        model: nn.Module,
        kv_cache: StaticKVCache,
        use_triton: bool = True,
    ):
        self.model     = model.eval()
        self.kv_cache  = kv_cache
        self.use_triton = use_triton

        if use_triton:
            patch_model(self.model, self.kv_cache, use_triton=True)

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
        """从 .pth checkpoint 加载模型并创建推理引擎。"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from model.model_minimind import MiniMindForCausalLM

        model = MiniMindForCausalLM(config)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device=device, dtype=dtype)
        model.eval()

        kv_cache = StaticKVCache.new(
            num_layers   = config.num_hidden_layers,
            num_kv_heads = config.num_key_value_heads,
            head_dim     = config.head_dim,
            max_batch    = 1,
            max_seq      = max_seq,
            device       = torch.device(device),
            dtype        = dtype,
        )
        return cls(model, kv_cache, use_triton=use_triton)

    # ── 推理主循环 ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,          # (1, T_prompt)
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.85,
        do_sample: bool = True,
        eos_token_id: int = 2,
        stream_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        """
        Args:
            input_ids:        prompt token ids，shape (1, T_prompt)
            max_new_tokens:   最大新生成 token 数
            temperature:      采样温度
            top_p:            nucleus 截断
            do_sample:        False = greedy
            eos_token_id:     遇到此 token 停止
            stream_callback:  每生成一个 token 调用，参数为 token_id（用于流式输出）

        Returns:
            generated_ids: List[int]，不含 prompt
        """
        self.kv_cache.reset()

        device = input_ids.device
        generated_ids: List[int] = []

        # ── Prefill ──────────────────────────────────────────────────────
        hidden, _, _ = self.model.model(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,   # 不走 HF cache，引擎自管理
        )
        # 推进 KV cache 指针
        self.kv_cache.step()

        # 取最后一个 token 的 logits
        logits = self.model.lm_head(hidden[:, -1:, :])   # (1, 1, vocab)
        next_token = sample(logits, temperature=temperature, top_p=top_p, do_sample=do_sample)
        generated_ids.append(next_token.item())
        if stream_callback:
            stream_callback(next_token.item())

        # ── Decode loop ───────────────────────────────────────────────────
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            cur_input = next_token.unsqueeze(0).to(device)   # (1, 1)

            hidden, _, _ = self.model.model(
                input_ids=cur_input,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
            )
            self.kv_cache.step()

            logits = self.model.lm_head(hidden[:, -1:, :])
            next_token = sample(logits, temperature=temperature, top_p=top_p, do_sample=do_sample)
            token_id = next_token.item()
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
