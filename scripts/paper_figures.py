#!/usr/bin/env python3
"""
Generate paper-ready statistics and tables.
Run after all experiments complete to produce the final paper numbers.
"""

import json
import os
import sys

RESULTS_BASE = "/nfs/hpc/share/zhanyaol/claude-code/results"


def load(path):
    full = os.path.join(RESULTS_BASE, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


# ============================================================
# TABLE 1: AFlow-Stream Main Results
# ============================================================

AFLOW_EXPERIMENTS = [
    ("Single-Agent",         "e18/single_agent_aflow_stream_seed42.json"),
    ("SC k=5",               "e10/self_consistency_aflow_stream_seed42.json"),
    ("AFlow† Qwen3-8B",      "e9b/aflow_aflow_stream_seed42.json"),
    ("DyLAN",                "e13/dylan_aflow_stream_seed42.json"),
    ("AgentNet",             "e35/agentnet_aflow_stream_seed42.json"),
    ("MemCollab",            "e33/memcollab_aflow_stream_seed42.json"),
    ("EvoMem",               "e37/evomem_aflow_stream_seed42.json"),
    ("EvoPool -CoDream",     "e15/evopool_no_codream_aflow_stream_seed42.json"),
    ("EvoPool +CoDream",     "e17/evopool_full_aflow_stream_seed42.json"),
]
AFLOW_DOMAINS = ["gsm8k", "hotpotqa", "mbpp", "math", "humaneval", "drop"]
AFLOW_SHORT   = {"gsm8k": "GSM", "hotpotqa": "HQA", "mbpp": "MBPP",
                 "math": "MATH", "humaneval": "HE", "drop": "DROP"}


def print_aflow_table():
    print("\n" + "="*95)
    print("TABLE 1: AFlow-Stream Results (Qwen3-8B backbone)")
    print("="*95)
    w = 26
    header = f"{'Method':<{w}}" + "".join(f" {AFLOW_SHORT[d]:>7}" for d in AFLOW_DOMAINS) + f" {'Mean':>7} {'AUC':>7}"
    print(header)
    print("-"*95)

    for name, path in AFLOW_EXPERIMENTS:
        data = load(path)
        if data is None:
            print(f"{name:<{w}}" + " {'—':>7}" * len(AFLOW_DOMAINS) + f" {'N/A':>7}")
            continue
        ds = data.get("domain_scores", {})
        s = data["summary"]
        cells = []
        for d in AFLOW_DOMAINS:
            v = ds.get(d, [])
            cells.append(f"{sum(v)/len(v):7.3f}" if v else f"{'—':>7}")
        print(f"{name:<{w}}" + "".join(cells) + f" {s['mean_score']:7.3f} {s.get('auc', 0):7.3f}")
    print("="*95)


# ============================================================
# TABLE 2: Hard Math Stream Main Results
# ============================================================

HARD_MATH_EXPERIMENTS = [
    ("Single-Agent",         "e39/single_agent_hard_math_stream_seed42.json"),
    ("AgentNet",             "e51/agentnet_hard_math_stream_seed42.json"),
    ("MemCollab",            "e52/memcollab_hard_math_stream_seed42.json"),
    ("EvoMem",               "e53/evomem_hard_math_stream_seed42.json"),
    ("DyLAN (fixed)",        "e54/dylan_hard_math_stream_seed42.json"),
    ("EvoPool -CoDream",     "e41/evopool_no_codream_hard_math_stream_seed42.json"),
    ("EvoPool +CoDream",     "e40/evopool_full_hard_math_stream_seed42.json"),
]
HM_DOMAINS = ["math_hard", "aime_2022", "aime_2023", "aime_2024", "aime_2025"]
HM_SHORT    = {"math_hard": "MATH-H", "aime_2022": "AIME22", "aime_2023": "AIME23",
               "aime_2024": "AIME24", "aime_2025": "AIME25"}


def load_hm(path):
    data = load(path)
    if data is None:
        return None
    tasks = data.get("per_task_results", [])
    if tasks:
        ds = {}
        for t in tasks:
            ds.setdefault(t.get("domain", "?"), []).append(t.get("score", 0))
        data["_ds"] = ds
    else:
        data["_ds"] = {}
    return data


def print_hardmath_table():
    print("\n" + "="*80)
    print("TABLE 2: Hard Math Stream Results (Qwen3-8B + thinking=True)")
    print("="*80)
    w = 22
    header = f"{'Method':<{w}}" + "".join(f" {HM_SHORT[d]:>8}" for d in HM_DOMAINS) + f" {'Mean':>7}"
    print(header)
    print("-"*80)

    for name, path in HARD_MATH_EXPERIMENTS:
        data = load_hm(path)
        if data is None:
            print(f"{name:<{w}}" + f" {'—':>8}" * len(HM_DOMAINS) + f" {'N/A':>7}")
            continue
        ds = data.get("_ds", {})
        means = []
        cells = []
        for d in HM_DOMAINS:
            v = ds.get(d, [])
            if v:
                m = sum(v) / len(v)
                cells.append(f"{m:8.3f}")
                means.append(m)
            else:
                cells.append(f"{'—':>8}")
        overall = sum(means) / len(means) if means else 0
        print(f"{name:<{w}}" + "".join(cells) + f" {overall:7.3f}")
    print("="*80)


# ============================================================
# ANALYSIS: Within-domain learning curves on AIME
# ============================================================

def print_aime_learning_curves():
    print("\n" + "="*80)
    print("AIME LEARNING CURVES: Within-domain B1→B3 trends (10-task bins)")
    print("="*80)

    conditions = [
        ("Single-Agent", "e39/single_agent_hard_math_stream_seed42.json"),
        ("EvoMem",       "e53/evomem_hard_math_stream_seed42.json"),
        ("MemCollab",    "e52/memcollab_hard_math_stream_seed42.json"),
        ("-CoDream",     "e41/evopool_no_codream_hard_math_stream_seed42.json"),
        ("+CoDream",     "e40/evopool_full_hard_math_stream_seed42.json"),
    ]

    for domain in ["aime_2022", "aime_2023", "aime_2024"]:
        print(f"\n  [{domain.upper()}]")
        print(f"  {'Condition':<16}" + "".join(f"  B{i+1:3}" for i in range(3)) + f"  {'Mean':>6}  {'Trend':>6}")
        for name, path in conditions:
            data = load_hm(path)
            if data is None:
                print(f"  {name:<16}  N/A")
                continue
            dt = [t for t in data.get("per_task_results", []) if t.get("domain") == domain]
            if not dt:
                print(f"  {name:<16}  (no tasks yet)")
                continue
            bins = [dt[i:i+10] for i in range(0, len(dt), 10)]
            bin_means = [sum(t["score"] for t in b)/len(b) for b in bins]
            while len(bin_means) < 3:
                bin_means.append(float("nan"))
            mean = sum(s for s in bin_means if s == s) / sum(1 for s in bin_means if s == s)
            trend = bin_means[-1] - bin_means[0] if bin_means[-1] == bin_means[-1] else 0.0
            row = f"  {name:<16}"
            for m in bin_means[:3]:
                row += f"  {m:.3f}" if m == m else "   N/A"
            row += f"  {mean:6.3f}  {trend:+6.3f}"
            print(row)


# ============================================================
# ANALYSIS: CoDream ablation on AFlow-Stream (for paper Table 3)
# ============================================================

def print_codream_ablation():
    ref_path = "e17/evopool_full_aflow_stream_seed42.json"
    ablations = [
        ("-CoDream (E15b)", "e15/evopool_no_codream_aflow_stream_seed42.json"),
        ("-Lifecycle (E19)", "e19/evopool_no_lifecycle_aflow_stream_seed42.json"),
        ("-L2 (E23)", "e23/evopool_no_l2_aflow_stream_seed42.json"),
        ("-L3 (E22)", "e22/evopool_no_l3_aflow_stream_seed42.json"),
        ("-Verify (E27)", "e27/evopool_no_verify_aflow_stream_seed42.json"),
        ("-Sym CD (E26)", "e26/evopool_symmetric_codream_aflow_stream_seed42.json"),
    ]

    ref = load(ref_path)
    if ref is None:
        return
    ref_ds = ref.get("domain_scores", {})
    ref_means = {d: sum(ref_ds[d])/len(ref_ds[d]) for d in AFLOW_DOMAINS if d in ref_ds}
    ref_overall = ref["summary"]["mean_score"]

    print("\n" + "="*95)
    print("TABLE 3: CoDream Ablation Gaps (delta vs EvoPool-full, AFlow-Stream)")
    print("="*95)
    w = 20
    header = f"{'Condition':<{w}}" + "".join(f" {AFLOW_SHORT[d]:>7}" for d in AFLOW_DOMAINS) + f" {'ΔMEAN':>7}"
    print(header)
    print("-"*95)
    ref_cells = " ".join(f"{ref_means.get(d, 0):7.3f}" for d in AFLOW_DOMAINS)
    print(f"{'EvoPool-full (ref)':<{w}} {ref_cells} {ref_overall:7.3f}")
    print("-"*95)

    for name, path in ablations:
        data = load(path)
        if data is None:
            continue
        ds = data.get("domain_scores", {})
        delta_cells = []
        for d in AFLOW_DOMAINS:
            v = ds.get(d, [])
            if v:
                diff = sum(v)/len(v) - ref_means.get(d, 0)
                delta_cells.append(f"{diff:+7.3f}")
            else:
                delta_cells.append(f"{'—':>7}")
        delta_overall = data["summary"]["mean_score"] - ref_overall
        print(f"{name:<{w}} " + " ".join(delta_cells) + f" {delta_overall:+7.3f}")
    print("="*95)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print_aflow_table()
    print_hardmath_table()
    print_codream_ablation()
    print_aime_learning_curves()
