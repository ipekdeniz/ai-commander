#!/usr/bin/env python3
"""
统计 Qwen3.5-27B 模型中 linear attention、softmax attention、MLP 的参数占比。
仅从 safetensors 元数据读取 shape，不加载完整张量。
"""
import json
import os
from pathlib import Path
from collections import defaultdict

# 尝试用 safetensors 的 get_slice 只读 shape（不加载数据）
try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def get_param_count_from_safetensors(path: str, weight_map: dict) -> dict[str, int]:
    """从各 shard 中按 name 读取 shape 并计算参数量，返回 name -> num_params。"""
    name_to_count = {}
    # 按文件分组 name
    file_to_names = defaultdict(list)
    for name, filename in weight_map.items():
        file_to_names[filename].append(name)

    base = Path(path)
    for filename, names in file_to_names.items():
        full_path = base / filename
        if not full_path.exists():
            continue
        with safe_open(full_path, framework="pt", device="cpu") as f:
            for name in names:
                if name not in f.keys():
                    continue
                try:
                    slice_obj = f.get_slice(name)
                    shape = slice_obj.get_shape()
                    n = 1
                    for d in shape:
                        n *= d
                    name_to_count[name] = n
                except Exception as e:
                    # 部分实现可能没有 get_slice，回退到加载
                    t = f.get_tensor(name)
                    name_to_count[name] = t.numel()
    return name_to_count


def classify_name(name: str) -> str | None:
    """
    将参数名分类为: linear_attn, softmax_attn, mlp。
    只统计 model.language_model.layers.* 下的这三类，其余返回 None（不纳入三类占比）。
    """
    if "model.language_model.layers." not in name:
        return None
    if ".linear_attn." in name:
        return "linear_attn"
    if ".self_attn." in name:
        return "softmax_attn"
    if ".mlp." in name:
        return "mlp"
    return None


def main():
    model_dir = Path(__file__).resolve().parent
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print("未找到 model.safetensors.index.json")
        return

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    if not HAS_SAFETENSORS:
        print("请安装 safetensors: pip install safetensors")
        return

    print("正在从 safetensors 读取 shape（不加载张量数据）...")
    name_to_count = get_param_count_from_safetensors(str(model_dir), weight_map)

    # 分类汇总
    total_lm = 0
    by_category = defaultdict(int)
    other = 0
    for name, count in name_to_count.items():
        kind = classify_name(name)
        if kind is not None:
            by_category[kind] += count
            total_lm += count
        else:
            # 只统计 language_model 内的其他参数（embed、norm 等），用于总和
            if "model.language_model." in name or name.startswith("lm_head."):
                other += count

    total_three = sum(by_category.values())
    total_all = total_three + other
    # 占比按“三类之和”为 100% 来算，便于看三者相对比例；同时给出占全模型比例
    print("\n========== 统计结果（仅 language_model 中 linear_attn / self_attn / mlp）==========\n")
    print(f"{'模块':<20} {'参数量':>18} {'占三类比例':>14} {'占全模型比例':>14}")
    print("-" * 70)
    for key, label in [
        ("linear_attn", "Linear Attention"),
        ("softmax_attn", "Softmax Attention"),
        ("mlp", "MLP"),
    ]:
        n = by_category[key]
        pct_three = 100.0 * n / total_three if total_three else 0
        pct_all = 100.0 * n / total_all if total_all else 0
        print(f"{label:<20} {n:>18,} {pct_three:>13.2f}% {pct_all:>13.2f}%")
    print("-" * 70)
    print(f"{'三类合计':<20} {total_three:>18,} {100.0:>13.2f}% {100.0 * total_three / total_all:>13.2f}%")
    print(f"{'其他(embed/norm等)':<20} {other:>18,} {'—':>14} {100.0 * other / total_all:>13.2f}%")
    print(f"{'全模型':<20} {total_all:>18,}")
    if "metadata" in index and "total_size" in index["metadata"]:
        reported = index["metadata"]["total_size"]
        print(f"{'index 中 total_size (字节)':<24} {reported:>18,}")
    print()


if __name__ == "__main__":
    main()
