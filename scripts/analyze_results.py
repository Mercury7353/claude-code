#!/usr/bin/env python3
"""Quick result analysis script for EvoPool experiments."""

import json
import os
import sys

RESULTS_BASE = "/nfs/hpc/share/zhanyaol/claude-code/results"

EXPERIMENTS = {
    "AFlow (E9)":             "e9/aflow_aflow_stream_seed42.json",
    "SC (E10)":               "e10/self_consistency_aflow_stream_seed42.json",
    "EvoPool full (E11)":     "e11/evopool_full_aflow_stream_seed42.json",
    "EvoPool fix1 (E12)":     "e12/evopool_full_aflow_stream_seed42.json",
    "DyLAN (E13)":            "e13/dylan_aflow_stream_seed42.json",
    "EvoPool noCoDream (E15)":"e15/evopool_no_codream_aflow_stream_seed42.json",
    "EvoPool fix1 (E11b)":    "e11b/evopool_full_aflow_stream_seed42.json",
    "EvoPool fix2 (E16)":     "e16/evopool_full_aflow_stream_seed42.json",
    "EvoPool noCoDream (E6)": "e6/evopool_no_codream_aflow_stream_seed42.json",
}

DOMAINS = ["gsm8k", "hotpotqa", "mbpp", "math", "humaneval", "drop"]
DOMAIN_SHORT = {"gsm8k": "GSM", "hotpotqa": "HQA", "mbpp": "MBPP",
                "math": "MATH", "humaneval": "HE", "drop": "DROP"}


def load(path):
    full = os.path.join(RESULTS_BASE, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def domain_mean(data, domain):
    scores = data.get("domain_scores", {}).get(domain, [])
    if not scores:
        return None
    return sum(scores) / len(scores), len(scores)


def print_table():
    print("\n" + "="*90)
    print("EvoPool Results Summary")
    print("="*90)
    header = f"{'Condition':<28} {'GSM':>6} {'HQA':>6} {'MBPP':>6} {'MATH':>6} {'HE':>6} {'DROP':>6} {'MEAN':>7} {'AUC':>7}"
    print(header)
    print("-"*90)

    for name, path in EXPERIMENTS.items():
        data = load(path)
        if data is None:
            print(f"{name:<28} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'N/A':>7} {'N/A':>7}")
            continue

        s = data["summary"]
        overall = s.get("mean_score", 0.0)
        auc = s.get("auc", 0.0)
        ds = data.get("domain_scores", {})

        cells = []
        for d in DOMAINS:
            scores = ds.get(d, [])
            if scores:
                cells.append(f"{sum(scores)/len(scores):6.3f}")
            else:
                cells.append(f"{'—':>6}")

        row = f"{name:<28} " + " ".join(cells) + f" {overall:7.3f} {auc:7.3f}"
        print(row)

    print("="*90)


def print_math_breakdown(exp_name, path, n_bins=5):
    data = load(path)
    if data is None:
        return
    tasks = data.get("per_task_results", [])
    math_tasks = [t for t in tasks if t.get("domain") == "math"]
    if not math_tasks:
        return

    bin_size = max(1, len(math_tasks) // n_bins)
    print(f"\nMATH domain breakdown for {exp_name}:")
    for i in range(0, len(math_tasks), bin_size):
        chunk = math_tasks[i:i + bin_size]
        mean = sum(t["score"] for t in chunk) / len(chunk)
        task_range = f"{chunk[0]['task_index']+1}-{chunk[-1]['task_index']+1}"
        print(f"  Tasks {task_range}: {mean:.3f} ({len(chunk)} tasks)")


def print_codream_comparison():
    e11 = load("e11/evopool_full_aflow_stream_seed42.json")
    e12 = load("e12/evopool_full_aflow_stream_seed42.json")
    if e11 and e12:
        print("\nFix1 impact (E12 vs E11, MATH domain breakdown):")
        for label, data in [("E11 (no fix1)", e11), ("E12 (fix1)", e12)]:
            tasks = data.get("per_task_results", [])
            math_tasks = [t for t in tasks if t.get("domain") == "math"]
            early = [t for t in math_tasks if t.get("task_index", 0) < 340]
            late = [t for t in math_tasks if t.get("task_index", 0) >= 340]
            e_mean = sum(t["score"] for t in early) / len(early) if early else 0
            l_mean = sum(t["score"] for t in late) / len(late) if late else 0
            print(f"  {label}: early(301-340)={e_mean:.3f} late(341-400)={l_mean:.3f}")


if __name__ == "__main__":
    print_table()
    print_math_breakdown("E11 EvoPool full", "e11/evopool_full_aflow_stream_seed42.json")
    print_math_breakdown("E12 fix1", "e12/evopool_full_aflow_stream_seed42.json")
    print_codream_comparison()
