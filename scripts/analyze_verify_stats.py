#!/usr/bin/env python3
"""
Analyze CoDream verify pass/reject statistics per domain.
Shows why CoDream benefits dependent tasks but not independent tasks.

Usage:
  python scripts/analyze_verify_stats.py results/e21/evopool_full_aflow_stream_seed42.json
  (E21 = E17 replica run with verify stats logging enabled)
"""
import json
import sys
from collections import defaultdict

DOMAIN_ORDER = ["gsm8k", "hotpotqa", "mbpp", "math", "humaneval", "drop"]
DEPENDENT = {"hotpotqa", "drop"}
INDEPENDENT = {"gsm8k", "mbpp", "math", "humaneval"}

def main(result_file: str):
    with open(result_file) as f:
        data = json.load(f)

    per_task = data.get("per_task_results", [])

    stats = defaultdict(lambda: {
        "n_tasks": 0, "n_codream_triggered": 0,
        "n_generated": 0, "n_verified": 0,
        "verify_rates": [],
    })

    for r in per_task:
        domain = r.get("domain", "unknown")
        s = stats[domain]
        s["n_tasks"] += 1
        gen = r.get("codream_generated", 0)
        ver = r.get("codream_verified", 0)
        if gen > 0:
            s["n_codream_triggered"] += 1
            s["n_generated"] += gen
            s["n_verified"] += ver
            s["verify_rates"].append(ver / gen)

    print("=" * 80)
    print("CoDream Verify Statistics by Domain")
    print("=" * 80)
    print(f"{'Domain':12s} {'Type':11s} {'Triggered':>9s} {'Generated':>9s} {'Verified':>8s} {'Rate':>6s} {'Acc':>6s}")
    print("-" * 80)

    for domain in DOMAIN_ORDER:
        s = stats[domain]
        if s["n_tasks"] == 0:
            continue
        domain_scores = data.get("domain_scores", {}).get(domain, [])
        acc = sum(domain_scores) / len(domain_scores) if domain_scores else 0.0
        n_trig = s["n_codream_triggered"]
        n_gen = s["n_generated"]
        n_ver = s["n_verified"]
        rate = n_ver / n_gen if n_gen > 0 else 0.0
        dtype = "DEPENDENT" if domain in DEPENDENT else "independent"
        print(f"{domain:12s} {dtype:11s} {n_trig:>9d} {n_gen:>9d} {n_ver:>8d} {rate:>5.1%} {acc:>6.3f}")

    print("=" * 80)
    print("\nSummary by task type:")
    for dtype, domains in [("DEPENDENT", DEPENDENT), ("independent", INDEPENDENT)]:
        total_gen = sum(stats[d]["n_generated"] for d in domains)
        total_ver = sum(stats[d]["n_verified"] for d in domains)
        rate = total_ver / total_gen if total_gen > 0 else 0.0
        print(f"  {dtype:11s}: {total_gen} generated, {total_ver} verified → {rate:.1%} pass rate")

    print("\nKey question: Do independent tasks have lower verify pass rates?")
    print("If yes → insights don't actually help independent tasks → explains ~0% gain")
    print("If no  → insights are verified but don't transfer to future tasks")

if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "results/e21/evopool_full_aflow_stream_seed42.json"
    main(f)
