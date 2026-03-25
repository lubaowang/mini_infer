#!/usr/bin/env python3
"""
infer.py
────────
MiniMind Triton 推理引擎入口，对标原项目的 eval.py。

用法：
    # 基础推理（Triton 加速，自动检测 GPU）
    python infer.py --load_from model --weight full_sft --hidden_size 512

    # 强制 PyTorch fallback（对比/调试用）
    python infer.py --load_from model --weight full_sft --backend torch

    # 自动测试模式（跑内置 prompts）
    python infer.py --load_from model --weight full_sft --input_mode 0

    # 性能 profiling（生成后打印算子耗时）
    python infer.py --load_from model --weight full_sft --profile

    # 算子 benchmark（不推理，只测速）
    python scripts/benchmark.py --hidden_size 512
"""

import argparse
import sys
import os
import time
import warnings
import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# 让 model/ engine/ kernels/ 都可以直接 import
sys.path.insert(0, os.path.dirname(__file__))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind Triton 推理引擎")

    # 模型配置（与原 eval.py 保持接口兼容）
    parser.add_argument("--load_from",   default="model", type=str,
                        help="模型来源：'model'=原生 .pth 权重，其他路径=HuggingFace 格式")
    parser.add_argument("--save_dir",    default="model",   type=str, help="权重目录")
    parser.add_argument("--weight",      default="full_sft", type=str,
                        help="权重名称前缀 (pretrain/full_sft/rlhf/reason)")
    parser.add_argument("--hidden_size", default=512,     type=int)
    parser.add_argument("--num_hidden_layers", default=16, type=int)
    parser.add_argument("--use_moe",     default=0,       type=int, choices=[0, 1])
    parser.add_argument("--inference_rope_scaling", default=False, action="store_true")

    # 推理超参
    parser.add_argument("--max_new_tokens", default=512,  type=int)
    parser.add_argument("--temperature",    default=0.85, type=float)
    parser.add_argument("--top_p",          default=0.85, type=float)
    parser.add_argument("--historys",       default=0,    type=int,
                        help="携带历史对话轮数（0=不携带）")

    # 引擎配置
    parser.add_argument("--backend",     default="triton", choices=["triton", "torch"],
                        help="triton=Triton 算子加速，torch=纯 PyTorch")
    parser.add_argument("--max_seq",     default=2048,  type=int, help="KV Cache 最大长度")
    parser.add_argument("--dtype",       default="fp16", choices=["fp16", "bf16", "fp32"])

    # 交互控制
    parser.add_argument("--input_mode",  default=-1,    type=int,
                        help="-1=启动时询问，0=自动测试，1=手动输入")
    parser.add_argument("--show_speed",  default=1,     type=int)
    parser.add_argument("--profile",     default=False, action="store_true",
                        help="生成完成后打印 torch.profiler 分析报告")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ─── 模型初始化 ───────────────────────────────────────────────────────────────

def init_engine(args):
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from engine import MiniMindInferenceEngine, StaticKVCache

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    use_triton = (args.backend == "triton") and torch.cuda.is_available()

    if use_triton:
        try:
            import triton  # noqa
        except ImportError:
            print("[warn] Triton 未安装，自动切换到 PyTorch 模式")
            use_triton = False

    # ── 加载 tokenizer ──────────────────────────────────────────────────────
    tok_path = args.load_from
    print(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # ── 构建模型 ────────────────────────────────────────────────────────────
    if "model" in args.load_from:
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling,
        )
        moe_suffix = "_moe" if args.use_moe else ""
        ckpt_path  = f"{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"

        print(f"[init] 加载权重: {ckpt_path}")
        model = MiniMindForCausalLM(config)
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu"), strict=True
        )
        model = model.to(device=args.device, dtype=dtype).eval()

        # 打印参数量
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[init] 参数量: {total_params:.1f}M")

    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        ).to(device=args.device, dtype=dtype).eval()
        config = model.config

    # ── 构建 KV Cache ──────────────────────────────────────────────────────
    kv_cache = StaticKVCache.new(
        num_layers   = config.num_hidden_layers,
        num_kv_heads = config.num_key_value_heads,
        head_dim     = config.head_dim,
        max_batch    = 1,
        max_seq      = args.max_seq,
        device       = torch.device(args.device),
        dtype        = dtype,
    )
    print(f"[init] KV Cache: {kv_cache}")

    # ── 构建推理引擎 ────────────────────────────────────────────────────────
    engine = MiniMindInferenceEngine(model, kv_cache, use_triton=use_triton)
    mode_str = "Triton 加速" if use_triton else "PyTorch 模式"
    print(f"[init] 推理后端: {mode_str}\n")

    return engine, tokenizer


# ─── 对话推理 ─────────────────────────────────────────────────────────────────

def run_chat(args, engine, tokenizer):
    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        "解释一下光合作用的基本过程",
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    if args.input_mode == -1:
        args.input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))

    conversation = []

    def stream_print(token_id: int):
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(text, end="", flush=True)

    prompt_iter = (
        iter(prompts) if args.input_mode == 0
        else iter(lambda: input("💬: "), "")
    )

    for prompt in prompt_iter:
        if args.input_mode == 0:
            print(f"💬: {prompt}")

        # 对话历史管理
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if args.weight == "reason":
            templates["enable_thinking"] = True

        if args.weight != "pretrain":
            input_text = tokenizer.apply_chat_template(**templates)
        else:
            input_text = tokenizer.bos_token + prompt

        inputs    = tokenizer(input_text, return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"]

        print("🤖: ", end="")
        st = time.perf_counter()

        if args.profile:
            # Profiling 模式：使用 torch.profiler
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                generated_ids = engine.generate(
                    input_ids         = input_ids,
                    max_new_tokens    = args.max_new_tokens,
                    temperature       = args.temperature,
                    top_p             = args.top_p,
                    eos_token_id      = tokenizer.eos_token_id,
                    stream_callback   = stream_print,
                )
            print()
            print("\n── Profiler Top-10 (CUDA time) ──")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            generated_ids = engine.generate(
                input_ids         = input_ids,
                max_new_tokens    = args.max_new_tokens,
                temperature       = args.temperature,
                top_p             = args.top_p,
                eos_token_id      = tokenizer.eos_token_id,
                stream_callback   = stream_print,
            )
            print()

        elapsed    = time.perf_counter() - st
        gen_tokens = len(generated_ids)
        response   = tokenizer.decode(generated_ids, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})

        if args.show_speed:
            print(f"[Speed] {gen_tokens / elapsed:.1f} tokens/s  "
                  f"({gen_tokens} tokens in {elapsed*1000:.0f}ms)\n\n")
        else:
            print()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    engine, tokenizer = init_engine(args)
    run_chat(args, engine, tokenizer)


if __name__ == "__main__":
    main()
