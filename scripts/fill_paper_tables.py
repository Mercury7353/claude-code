#!/usr/bin/env python3
"""
Reads experiment result files and prints filled paper tables.
Run after E19/E22/E23/E24/E9b complete.
"""
import json
import os

RESULTS_BASE = "/nfs/hpc/share/zhanyaol/claude-code/results"
DOMAINS = ["gsm8k", "hotpotqa", "mbpp", "math", "humaneval", "drop"]
SHORT = {"gsm8k": "GSM", "hotpotqa": "HQA", "mbpp": "MBPP", "math": "MATH", "humaneval": "HE", "drop": "DROP"}


def load(path):
    full = os.path.join(RESULTS_BASE, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def dmeans(data):
    ds = data.get("domain_scores", {})
    return {d: (sum(ds[d])/len(ds[d]) if ds.get(d) else None) for d in DOMAINS}


def mean_str(v):
    return f"{v:.3f}" if v is not None else "[TBD]"


def row(name, path):
    data = load(path)
    if data is None:
        cells = " | ".join(["[TBD]"] * 6)
        return f"| {name} | {cells} | [TBD] |"
    dm = dmeans(data)
    cells = " | ".join(mean_str(dm[d]) for d in DOMAINS)
    overall = mean_str(data["summary"]["mean_score"])
    return f"| {name} | {cells} | {overall} |"


def first_n_mean(data, domain, n=5):
    tasks = data.get("per_task_results", [])
    domain_tasks = [t for t in tasks if t.get("domain") == domain]
    chunk = domain_tasks[:n]
    if not chunk:
        return None
    return sum(t["score"] for t in chunk) / len(chunk)


def quartile(data, domain, q):
    """q=1: first 25, q=4: last 25"""
    tasks = data.get("per_task_results", [])
    domain_tasks = [t for t in tasks if t.get("domain") == domain]
    if len(domain_tasks) < 25:
        return None
    if q == 1:
        chunk = domain_tasks[:25]
    else:
        chunk = domain_tasks[75:100] if len(domain_tasks) >= 100 else domain_tasks[-25:]
    return sum(t["score"] for t in chunk) / len(chunk)


# ===== Experiments =====
MAIN_TABLE = [
    ("Single-Agent (E18)",       "e18/single_agent_aflow_stream_seed42.json"),
    ("SC k=5 (E10)",             "e10/self_consistency_aflow_stream_seed42.json"),
    ("AFlow fixed (E9b)",        "e9b/aflow_aflow_stream_seed42.json"),
    ("DyLAN (E13)",              "e13/dylan_aflow_stream_seed42.json"),
    ("EvoPool -CoDream (E15b)",  "e15/evopool_no_codream_aflow_stream_seed42.json"),
    ("EvoPool +CoDream (E17)",   "e17/evopool_full_aflow_stream_seed42.json"),
]

ABLATION_TABLE = [
    ("EvoPool -CoDream (E15b)", "e15/evopool_no_codream_aflow_stream_seed42.json"),
    ("EvoPool -L3 (E22)",       "e22/evopool_no_l3_aflow_stream_seed42.json"),
    ("EvoPool -L2 (E23)",       "e23/evopool_no_l2_aflow_stream_seed42.json"),
    ("EvoPool +CoDream (E17)",  "e17/evopool_full_aflow_stream_seed42.json"),
    ("EvoPool -Lifecycle (E19)","e19/evopool_no_lifecycle_aflow_stream_seed42.json"),
    ("EvoPool RandTeam (E24)",  "e24/evopool_random_team_aflow_stream_seed42.json"),
]


def print_main_table():
    print("\n## Table 1: Main Results")
    print("| Method | GSM | HQA | MBPP | MATH | HE | DROP | Mean |")
    print("|--------|-----|-----|------|------|----|------|------|")
    for name, path in MAIN_TABLE:
        print(row(name, path))


def print_ablation_table():
    print("\n## Table 2: CoDream Component Ablations")
    print("| Method | GSM | HQA | MBPP | MATH | HE | DROP | Mean |")
    print("|--------|-----|-----|------|------|----|------|------|")
    for name, path in ABLATION_TABLE:
        print(row(name, path))


def print_quartile_table():
    print("\n## Table 3: Q1/Q4 Within-Domain Learning")
    exps = [
        ("E17 +CoDream",   "e17/evopool_full_aflow_stream_seed42.json"),
        ("E15b -CoDream",  "e15/evopool_no_codream_aflow_stream_seed42.json"),
        ("E22 -L3",        "e22/evopool_no_l3_aflow_stream_seed42.json"),
        ("E23 -L2",        "e23/evopool_no_l2_aflow_stream_seed42.json"),
    ]
    for domain in ["hotpotqa", "drop", "math"]:
        print(f"\n### {domain.upper()}")
        print("| Method | Q1 | Q4 | Trend | First-5 |")
        print("|--------|----|----|-------|---------|")
        for name, path in exps:
            data = load(path)
            if data is None:
                print(f"| {name} | [TBD] | [TBD] | [TBD] | [TBD] |")
                continue
            q1 = quartile(data, domain, 1)
            q4 = quartile(data, domain, 4)
            f5 = first_n_mean(data, domain, 5)
            trend = (q4 - q1) if (q1 is not None and q4 is not None) else None
            trend_str = f"{trend:+.3f}" if trend is not None else "[TBD]"
            print(f"| {name} | {mean_str(q1)} | {mean_str(q4)} | {trend_str} | {mean_str(f5)} |")


if __name__ == "__main__":
    print_main_table()
    print_ablation_table()
    print_quartile_table()
