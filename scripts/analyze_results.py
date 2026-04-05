#!/usr/bin/env python3
"""Quick result analysis script for EvoPool experiments."""

import json
import os
import sys

RESULTS_BASE = "/nfs/hpc/share/zhanyaol/claude-code/results"

EXPERIMENTS = {
    # --- Main table (paper) ---
    "Single-Agent (E18)":        "e18/single_agent_aflow_stream_seed42.json",
    "SC k=5 (E10)":              "e10/self_consistency_aflow_stream_seed42.json",
    "AFlow† fixed (E9b)":        "e9b/aflow_aflow_stream_seed42.json",
    "DyLAN (E13)":               "e13/dylan_aflow_stream_seed42.json",
    "EvoPool-noCoDream (E15b)":  "e15/evopool_no_codream_aflow_stream_seed42.json",
    "EvoPool-full (E17)":        "e17/evopool_full_aflow_stream_seed42.json",
    # --- Ablations (paper section 4.3–4.6) ---
    "EvoPool-noLifecycle (E19)": "e19/evopool_no_lifecycle_aflow_stream_seed42.json",
    "EvoPool-noL3 (E22)":        "e22/evopool_no_l3_aflow_stream_seed42.json",
    "EvoPool-noL2 (E23)":        "e23/evopool_no_l2_aflow_stream_seed42.json",
    "EvoPool-randTeam (E24)":    "e24/evopool_random_team_aflow_stream_seed42.json",
    # --- Enhanced CoDream (E25) ---
    "EvoPool-enh_codream (E25)": "e25/evopool_enhanced_codream_aflow_stream_seed42.json",
    # --- Reference / older ---
    "AFlow (E9, buggy fmt)":     "e9/aflow_aflow_stream_seed42.json",
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


def quartile_trends(domains=("hotpotqa", "drop", "math")):
    """Print Q1/Q4 accuracy trends for specified domains across all experiments."""
    experiments_to_show = {
        "E17 +CoDream":   "e17/evopool_full_aflow_stream_seed42.json",
        "E15b -CoDream":  "e15/evopool_no_codream_aflow_stream_seed42.json",
        "E19 -Lifecycle": "e19/evopool_no_lifecycle_aflow_stream_seed42.json",
        "E22 -L3":        "e22/evopool_no_l3_aflow_stream_seed42.json",
        "E23 -L2":        "e23/evopool_no_l2_aflow_stream_seed42.json",
        "E24 RandTeam":   "e24/evopool_random_team_aflow_stream_seed42.json",
        "E18 Single":     "e18/single_agent_aflow_stream_seed42.json",
    }

    print("\n" + "="*80)
    print("Q1/Q4 Within-Domain Learning Trends")
    print("="*80)

    for domain in domains:
        print(f"\n--- {domain.upper()} ---")
        header = f"  {'Exp':<18} {'Q1 (1-25)':>12} {'Q4 (76-100)':>12} {'Trend':>8} {'first5':>8}"
        print(header)
        for name, path in experiments_to_show.items():
            data = load(path)
            if data is None:
                print(f"  {name:<18} {'N/A':>12}")
                continue
            tasks = data.get("per_task_results", [])
            domain_tasks = [t for t in tasks if t.get("domain") == domain]
            if len(domain_tasks) < 25:
                print(f"  {name:<18} {'<25 tasks':>12}")
                continue
            q1 = domain_tasks[:25]
            q4 = domain_tasks[75:100] if len(domain_tasks) >= 100 else domain_tasks[-25:]
            first5 = domain_tasks[:5]
            q1_mean = sum(t["score"] for t in q1) / len(q1)
            q4_mean = sum(t["score"] for t in q4) / len(q4) if q4 else 0
            first5_mean = sum(t["score"] for t in first5) / len(first5) if first5 else 0
            trend = q4_mean - q1_mean
            trend_str = f"{trend:+.3f}"
            print(f"  {name:<18} {q1_mean:12.3f} {q4_mean:12.3f} {trend_str:>8} {first5_mean:8.3f}")

    print("="*80)


def codream_gains():
    """Print CoDream ablation: E17 vs E15b per domain, and new ablations vs E17."""
    ref_path = "e17/evopool_full_aflow_stream_seed42.json"
    comparisons = [
        ("E15b -CoDream", "e15/evopool_no_codream_aflow_stream_seed42.json"),
        ("E19 -Lifecycle", "e19/evopool_no_lifecycle_aflow_stream_seed42.json"),
        ("E22 -L3",        "e22/evopool_no_l3_aflow_stream_seed42.json"),
        ("E23 -L2",        "e23/evopool_no_l2_aflow_stream_seed42.json"),
        ("E24 RandTeam",   "e24/evopool_random_team_aflow_stream_seed42.json"),
    ]

    ref = load(ref_path)
    if ref is None:
        print("E17 reference not found")
        return

    print("\n" + "="*90)
    print("Ablation Gaps (vs E17 EvoPool-full)")
    print("="*90)
    header = f"{'Ablation':<22} {'GSM':>6} {'HQA':>6} {'MBPP':>6} {'MATH':>6} {'HE':>6} {'DROP':>6} {'MEAN':>7}"
    print(header)
    print("-"*90)

    ref_ds = ref.get("domain_scores", {})
    ref_means = {d: sum(ref_ds[d])/len(ref_ds[d]) for d in DOMAINS if d in ref_ds}
    ref_overall = ref["summary"]["mean_score"]
    ref_cells = " ".join(f"{ref_means.get(d, 0):6.3f}" for d in DOMAINS)
    print(f"{'E17 +full (ref)':<22} {ref_cells} {ref_overall:7.3f}")

    for name, path in comparisons:
        data = load(path)
        if data is None:
            print(f"{name:<22} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'N/A':>7}")
            continue
        ds = data.get("domain_scores", {})
        cells = []
        for d in DOMAINS:
            scores = ds.get(d, [])
            if scores:
                val = sum(scores)/len(scores)
                diff = val - ref_means.get(d, 0)
                cells.append(f"{diff:+6.3f}")
            else:
                cells.append(f"{'—':>6}")
        overall_diff = data["summary"]["mean_score"] - ref_overall
        print(f"{name:<22} " + " ".join(cells) + f" {overall_diff:+7.3f}")

    print("="*90)


def analyze_e25_vs_e17():
    """Compare E25 enhanced CoDream vs E17 standard, focusing on independent tasks."""
    e17 = load("e17/evopool_full_aflow_stream_seed42.json")
    e25 = load("e25/evopool_enhanced_codream_aflow_stream_seed42.json")

    if e17 is None:
        print("E17 not found")
        return
    if e25 is None:
        print("E25 not yet complete")
        return

    print("\n" + "="*80)
    print("E25 Enhanced CoDream vs E17 Standard CoDream")
    print("Focus: Independent tasks (MATH/GSM8K/MBPP/HE) — target of E25 improvements")
    print("="*80)

    def q_trend(data, domain):
        tasks = [t for t in data.get("per_task_results", []) if t.get("domain") == domain]
        if len(tasks) < 25: return None, None, None
        q1 = sum(t["score"] for t in tasks[:25]) / 25
        q4 = sum(t["score"] for t in tasks[75:100]) / 25 if len(tasks) >= 100 else sum(t["score"] for t in tasks[-25:]) / 25
        return q1, q4, q4 - q1

    print(f"\n{'Domain':12s}  {'E17 Q1':>8} {'E17 Q4':>8} {'E17 Δ':>8}  {'E25 Q1':>8} {'E25 Q4':>8} {'E25 Δ':>8}  {'Mean Δ':>8}")
    for dom in DOMAINS:
        e17_q1, e17_q4, e17_t = q_trend(e17, dom)
        e25_q1, e25_q4, e25_t = q_trend(e25, dom)
        e17_mean = sum(e17["domain_scores"].get(dom,[0]))/max(1,len(e17["domain_scores"].get(dom,[1])))
        e25_mean = sum(e25["domain_scores"].get(dom,[0]))/max(1,len(e25["domain_scores"].get(dom,[1])))
        mean_delta = e25_mean - e17_mean
        if e17_q1 is not None and e25_q1 is not None:
            print(f"{dom:12s}  {e17_q1:8.3f} {e17_q4:8.3f} {e17_t:+8.3f}  {e25_q1:8.3f} {e25_q4:8.3f} {e25_t:+8.3f}  {mean_delta:+8.3f}")
        else:
            print(f"{dom:12s}  {'N/A':>8}")

    e17_overall = e17["summary"]["mean_score"]
    e25_overall = e25["summary"]["mean_score"]
    print(f"\nOverall: E17={e17_overall:.3f}  E25={e25_overall:.3f}  Δ={e25_overall-e17_overall:+.3f}")

    # CoDream stats from E25 (has verify stats)
    print("\n--- CoDream trigger stats (E25, from per_task_results) ---")
    from collections import defaultdict
    dom_stats = defaultdict(lambda: {"tasks": 0, "triggered": 0, "gen": 0, "ver": 0})
    for t in e25.get("per_task_results", []):
        dom = t.get("domain", "unk")
        dom_stats[dom]["tasks"] += 1
        gen = t.get("codream_generated", 0)
        ver = t.get("codream_verified", 0)
        dom_stats[dom]["gen"] += gen
        dom_stats[dom]["ver"] += ver
        if gen > 0:
            dom_stats[dom]["triggered"] += 1
    print(f"{'Domain':12s}  {'tasks':>6} {'triggered':>10} {'trig%':>7} {'gen':>6} {'ver':>6} {'ver%':>7}")
    for dom in DOMAINS:
        s = dom_stats[dom]
        if s["tasks"] == 0: continue
        trate = s["triggered"] / s["tasks"]
        vrate = s["ver"] / s["gen"] if s["gen"] > 0 else 0
        print(f"{dom:12s}  {s['tasks']:6d} {s['triggered']:10d} {trate:7.2f} {s['gen']:6d} {s['ver']:6d} {vrate:7.2f}")

    print("="*80)


if __name__ == "__main__":
    print_table()
    quartile_trends()
    codream_gains()
    analyze_e25_vs_e17()
