"""
engine/kv_cache.py  （修正版）
──────────────────
静态预分配的 KV-Cache 管理器。

核心修正（相对原版）：
  原版将 update() 暂存到 _pending，等 step() 才真正写 buffer，
  导致 Attention 在 forward 中调用 get() 时拿不到本步新算的 K/V。

  新设计：
    update_and_get(layer_idx, new_k, new_v)
      → 立即写入 buffer[committed_len : committed_len+T_new]
      → 立即返回 buffer[0 : committed_len+T_new] 的完整切片供 Attention 使用
    step(T_new)
      → 一次 forward 所有层都跑完后调用，推进 committed_len
    reset()
      → 新请求时重置指针
"""

import torch
from typing import List, Tuple


class StaticKVCache:
    def __init__(
        self,
        buffers: List[Tuple[torch.Tensor, torch.Tensor]],
        max_seq: int,
    ):
        self.buffers       = buffers
        self.max_seq       = max_seq
        self.committed_len = 0

    @classmethod
    def new(
        cls,
        num_layers:   int,
        num_kv_heads: int,
        head_dim:     int,
        max_batch:    int = 1,
        max_seq:      int = 2048,
        device:       torch.device = torch.device("cuda"),
        dtype:        torch.dtype = torch.float16,
    ) -> "StaticKVCache":
        buffers = []
        for _ in range(num_layers):
            k_buf = torch.zeros(max_batch, num_kv_heads, max_seq, head_dim,
                                device=device, dtype=dtype)
            v_buf = torch.zeros(max_batch, num_kv_heads, max_seq, head_dim,
                                device=device, dtype=dtype)
            buffers.append((k_buf, v_buf))
        return cls(buffers, max_seq)

    def update_and_get(
        self,
        layer_idx: int,
        new_k: torch.Tensor,   # (B, Hk, T_new, D)
        new_v: torch.Tensor,   # (B, Hk, T_new, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        写入新 K/V 到 buffer，返回 [0 : committed_len+T_new] 的完整有效切片。
        committed_len 在此时还未推进（等 step() 后才推进）。
        """
        T_new = new_k.shape[2]
        write_end = self.committed_len + T_new
        assert write_end <= self.max_seq, (
            f"KV cache 溢出: committed={self.committed_len}, "
            f"T_new={T_new}, max_seq={self.max_seq}"
        )
        k_buf, v_buf = self.buffers[layer_idx]
        k_buf[:, :, self.committed_len:write_end, :] = new_k
        v_buf[:, :, self.committed_len:write_end, :] = new_v
        return k_buf[:, :, :write_end, :], v_buf[:, :, :write_end, :]

    def step(self, T_new: int) -> None:
        """forward 完成后推进指针。T_new = prefill 长度 or 1（decode）。"""
        self.committed_len += T_new
        assert self.committed_len <= self.max_seq

    def reset(self) -> None:
        self.committed_len = 0

    @property
    def is_empty(self) -> bool:
        return self.committed_len == 0

    def memory_bytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in self.buffers)

    def __repr__(self) -> str:
        mb = self.memory_bytes() / 1024 ** 2
        return (
            f"StaticKVCache(layers={len(self.buffers)}, "
            f"committed={self.committed_len}/{self.max_seq}, "
            f"mem={mb:.1f} MB)"
        )
