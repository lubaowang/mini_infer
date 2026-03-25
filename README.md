# MiniMind Triton/CUDA 推理工程

基于 MiniMind（Qwen3 架构改进版）的手写高性能推理引擎，包含 Triton 算子实现和完整的推理 pipeline。

## 工程结构

```
minimind_infer/
├── kernels/                    # 手写算子
│   ├── __init__.py
│   ├── rms_norm.py             # Triton RMSNorm（含 fp32 acc）
│   ├── rope_emb.py             # Triton RoPE（fused cos/sin apply）
│   ├── flash_attn_triton.py    # Triton Flash Attention（causal）
│   ├── silu_gate.py            # Triton Fused SiLU-Gate（SwiGLU MLP）
│   └── gemv.py                 # Triton GEMV（decode 阶段优化）
├── engine/                     # 推理引擎
│   ├── __init__.py
│   ├── kv_cache.py             # 静态 KV-Cache 管理（PagedAttention 简化版）
│   ├── sampler.py              # 采样器（Top-P / Temperature / Greedy）
│   └── inference_engine.py     # 主引擎（集成 Triton 算子替换）
├── model/
│   └── model_minimind.py       # 模型定义（直接 symlink 或 copy）
├── scripts/
│   ├── convert_weights.py      # 将原始 .pth 权重转为引擎格式
│   └── benchmark.py            # 性能基准测试（Triton vs PyTorch）
├── tests/
│   ├── test_kernels.py         # 算子正确性验证（对比 PyTorch 参考实现）
│   └── test_inference.py       # 端到端推理测试
├── benchmarks/
│   └── profile_ops.py          # nsys / torch.profiler 集成
├── configs/
│   └── default.yaml            # 推理默认配置
├── infer.py                    # 入口（对标原 eval.py）
├── requirements.txt
└── setup.py
```

## 快速开始

### 1. 环境安装
```bash
# CUDA >= 11.8, Python >= 3.10
pip install -r requirements.txt

# 验证 Triton 算子正确性
python -m pytest tests/test_kernels.py -v

# 运行 benchmark
python scripts/benchmark.py --hidden_size 512 --seq_len 1024
```

### 2. 推理
```bash
# 使用 Triton 算子（默认）
python infer.py --load_from model --weight full_sft --hidden_size 512

# 强制 PyTorch fallback（对比用）
python infer.py --load_from model --weight full_sft --hidden_size 512 --backend torch

# 性能 profiling
python infer.py --load_from model --weight full_sft --profile
```

## 算子说明

| 算子 | 对应操作 | 优化策略 |
|------|---------|---------|
| `rms_norm` | RMSNorm（含 q_norm/k_norm） | fp32 累加，向量化 load/store |
| `rope_emb` | Rotary Position Embedding | fuse cos*q + sin*rotate(q) |
| `flash_attn_triton` | Causal Self-Attention | tiled softmax，online normalization |
| `silu_gate` | SwiGLU: down(silu(gate)*up) | 三矩阵 fuse gate+silu+mul |
| `gemv` | decode 阶段 token×weight | warp-level reduction |

## 环境要求

- CUDA >= 11.8（推荐 12.x）
- PyTorch >= 2.1
- Triton >= 2.2（随 PyTorch 安装，或 `pip install triton`）
- GPU: NVIDIA Ampere 架构及以上（A10/A100/4090 等，sm_80+）
