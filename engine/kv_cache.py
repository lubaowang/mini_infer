"""
engine/kv_cache.py
──────────────────
静态预分配的 KV-Cache 管理器。

设计目标：
  - 启动时一次性分配 max_batch × max_seq × num_layers × 2 × kv_heads × head_dim 的显存
  - 推理时通过 seq_len 指针追加，避免反复 torch.cat（动态增长方式有大量内存拷贝）
  - 接口与 MiniMind 原始 past_key_values（List[Tuple]）兼容

结构：
  cache[layer_idx] = (
      K_buf: (max_batch, kv_heads, max_seq, head_dim),
      V_buf: (max_batch, kv_heads, max_seq, head_dim),
  )
  cur_len: int，记录当前已填充的 token 数
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class StaticKVCache:
    """
    推理引擎使用的静态 KV-Cache。

    使用方式：
        cache = StaticKVCache.new(config, max_batch, max_seq, device)
        # 逐层填入：
        cache.update(layer_idx, new_k, new_v)
        # 取当前有效 KV：
        k, v = cache.get(layer_idx)
        # 生成下一个 token 后：
        cache.step()
        # 新请求时清零：
        cache.reset()
    """

    # 每层存储：(K_buf, V_buf)，各 (B, Hk, max_seq, D)
    buffers: List[Tuple[torch.Tensor, torch.Tensor]]
    cur_len: int = 0
    max_seq: int = 2048
    _pending_k: List[Optional[torch.Tensor]] = field(default_factory=list)
    _pending_v: List[Optional[torch.Tensor]] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch: int = 1,
        max_seq: int = 2048,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ) -> "StaticKVCache":
        buffers = []
        for _ in range(num_layers):
            k_buf = torch.zeros(max_batch, num_kv_heads, max_seq, head_dim,
                                device=device, dtype=dtype)
            v_buf = torch.zeros(max_batch, num_kv_heads, max_seq, head_dim,
                                device=device, dtype=dtype)
            buffers.append((k_buf, v_buf))

        pending_k = [None] * num_layers
        pending_v = [None] * num_layers
        obj = cls.__new__(cls)
        obj.buffers  = buffers
        obj.cur_len  = 0
        obj.max_seq  = max_seq
        obj._pending_k = pending_k
        obj._pending_v = pending_v
        return obj

    # ── 填入当前 step 的 KV ────────────────────────────────────────────────
    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,   # (B, Hk, T_new, D)
        new_v: torch.Tensor,   # (B, Hk, T_new, D)
    ) -> None:
        """将新产生的 K/V 暂存（等 step() 时统一写入 buffer 并推进指针）。"""
        self._pending_k[layer_idx] = new_k
        self._pending_v[layer_idx] = new_v

    def step(self) -> None:
        """
        将所有层的 pending K/V 写入 buffer，推进 cur_len。
        在一个完整的前向传播完成后调用一次。
        """
        if self._pending_k[0] is None:
            return
        T_new = self._pending_k[0].shape[2]
        end   = self.cur_len + T_new
        assert end <= self.max_seq, (
            f"KV cache 溢出：cur_len={self.cur_len}, T_new={T_new}, max_seq={self.max_seq}"
        )
        for i, (k_buf, v_buf) in enumerate(self.buffers):
            k_buf[:, :, self.cur_len:end, :] = self._pending_k[i]
            v_buf[:, :, self.cur_len:end, :] = self._pending_v[i]
            self._pending_k[i] = None
            self._pending_v[i] = None
        self.cur_len = end

    # ── 取当前有效 KV 切片 ─────────────────────────────────────────────────
    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 layer_idx 层已填入的有效 KV，供 Attention 使用：
          K: (B, Hk, cur_len, D)
          V: (B, Hk, cur_len, D)
        """
        k_buf, v_buf = self.buffers[layer_idx]
        return k_buf[:, :, :self.cur_len, :], v_buf[:, :, :self.cur_len, :]

    # ── 重置（新 prompt） ───────────────────────────────────────────────────
    def reset(self) -> None:
        self.cur_len = 0
        # 不清零显存，后续写入会覆盖，省去清零开销

    # ── 状态查询 ────────────────────────────────────────────────────────────
    @property
    def is_empty(self) -> bool:
        return self.cur_len == 0

    def memory_bytes(self) -> int:
        total = 0
        for k_buf, v_buf in self.buffers:
            total += k_buf.nbytes + v_buf.nbytes
        return total

    def __repr__(self) -> str:
        mb = self.memory_bytes() / 1024 ** 2
        return (
            f"StaticKVCache("
            f"layers={len(self.buffers)}, "
            f"cur_len={self.cur_len}/{self.max_seq}, "
            f"memory={mb:.1f} MB)"
        )
