#!/usr/bin/env python3
"""
E25 vs E17: Deep analysis of enhanced CoDream on independent tasks.

Compares:
  E17: standard CoDream (threshold-only trigger, failure-only crystallize, no domain_general)
  E25: enhanced CoDream (disagreement trigger + success extraction + domain_general tier)

Key questions:
  1. How much more often does CoDream trigger in E25 vs E17?
  2. What is the verify pass rate per domain in E25?
  3. Do MATH/GSM8K/MBPP/HE show improved Q1→Q4 learning trend in E25?
  4. Does overall accuracy improve on independent tasks?
"""
import json
import sys
from collections import defaultdict

DOMAIN_ORDER = ["gsm8k", "hotpotqa", "mbpp", "math", "humaneval", "drop"]
DEPENDENT = {"hotpotqa", "drop"}
INDEPENDENT = {"gsm8k", "mbpp", "math", "humaneval"}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def quartile_stats(tasks, domain):
    dom_tasks = [t for t in tasks if t.get("domain") == domain]
    if len(dom_tasks) < 25:
        return None
    q1 = sum(t["score"] for t in dom_tasks[:25]) / 25
    q4 = sum(t["score"] for t in dom_tasks[75:100]) / 25 if len(dom_tasks) >= 100 else sum(t["score"] for t in dom_tasks[-25:]) / 25
    mean = sum(t["score"] for t in dom_tasks) / len(dom_tasks)
    return {"q1": q1, "q4": q4, "trend": q4 - q1, "mean": mean}


def codream_stats(tasks, domain):
    dom_tasks = [t for t in tasks if t.get("domain") == domain]
    n = len(dom_tasks)
    triggered = sum(1 for t in dom_tasks if t.get("codream_generated", 0) > 0)
    gen = sum(t.get("codream_generated", 0) for t in dom_tasks)
    ver = sum(t.get("codream_verified", 0) for t in dom_tasks)
    ver_rate = ver / gen if gen > 0 else 0.0
    trig_rate = triggered / n if n > 0 else 0.0
    return {"n": n, "triggered": triggered, "trig_rate": trig_rate, "gen": gen, "ver": ver, "ver_rate": ver_rate}


def main():
    e17_path = "results/e17/evopool_full_aflow_stream_seed42.json"
    e25_path = "results/e25/evopool_enhanced_codream_aflow_stream_seed42.json"

    e17 = load(e17_path)
    e25 = load(e25_path)

    if e17 is None:
        print("ERROR: E17 not found")
        return
    if e25 is None:
        print("E25 not yet complete — run after job 20141057 finishes")
        return

    e17_tasks = e17.get("per_task_results", [])
    e25_tasks = e25.get("per_task_results", [])

    print("=" * 90)
    print("E25 Enhanced CoDream vs E17 Standard CoDream")
    print("=" * 90)

    # 1. Domain accuracy comparison
    print("\n--- 1. Domain accuracy comparison ---")
    print(f"{'Domain':12s} {'Type':11s} {'E17 mean':>9s} {'E25 mean':>9s} {'Delta':>7s}")
    for dom in DOMAIN_ORDER:
        e17_q = quartile_stats(e17_tasks, dom)
        e25_q = quartile_stats(e25_tasks, dom)
        dtype = "DEPENDENT" if dom in DEPENDENT else "independent"
        if e17_q and e25_q:
            delta = e25_q["mean"] - e17_q["mean"]
            print(f"{dom:12s} {dtype:11s} {e17_q['mean']:9.3f} {e25_q['mean']:9.3f} {delta:+7.3f}")
        else:
            e17_mean = sum(e17["domain_scores"].get(dom, [0])) / max(1, len(e17["domain_scores"].get(dom, [1])))
            e25_mean = sum(e25["domain_scores"].get(dom, [0])) / max(1, len(e25["domain_scores"].get(dom, [1])))
            delta = e25_mean - e17_mean
            print(f"{dom:12s} {dtype:11s} {e17_mean:9.3f} {e25_mean:9.3f} {delta:+7.3f}")

    e17_overall = e17["summary"]["mean_score"]
    e25_overall = e25["summary"]["mean_score"]
    print(f"\nOverall: E17={e17_overall:.3f}  E25={e25_overall:.3f}  Δ={e25_overall - e17_overall:+.3f}")

    # 2. Q1/Q4 learning trend comparison
    print("\n--- 2. Within-domain learning trend Q1→Q4 ---")
    print(f"{'Domain':12s} {'Type':11s} {'E17 Q1':>8s} {'E17 Q4':>8s} {'E17 Δ':>7s}  {'E25 Q1':>8s} {'E25 Q4':>8s} {'E25 Δ':>7s}  {'Δ trend':>8s}")
    for dom in DOMAIN_ORDER:
        e17_q = quartile_stats(e17_tasks, dom)
        e25_q = quartile_stats(e25_tasks, dom)
        dtype = "DEPENDENT" if dom in DEPENDENT else "independent"
        if e17_q and e25_q:
            delta_trend = e25_q["trend"] - e17_q["trend"]
            print(f"{dom:12s} {dtype:11s} {e17_q['q1']:8.3f} {e17_q['q4']:8.3f} {e17_q['trend']:+7.3f}  {e25_q['q1']:8.3f} {e25_q['q4']:8.3f} {e25_q['trend']:+7.3f}  {delta_trend:+8.3f}")
        else:
            print(f"{dom:12s} {dtype:11s} {'N/A':>8s}")

    # 3. CoDream trigger rate comparison (E25 has stats, E17 doesn't — estimate E17 from score dist)
    print("\n--- 3. CoDream trigger statistics (E25 only, E17 lacked logging) ---")
    print(f"{'Domain':12s} {'Type':11s} {'tasks':>6s} {'triggered':>10s} {'trig%':>7s} {'gen':>6s} {'ver':>6s} {'ver%':>7s}")
    for dom in DOMAIN_ORDER:
        cs = codream_stats(e25_tasks, dom)
        dtype = "DEPENDENT" if dom in DEPENDENT else "independent"
        print(f"{dom:12s} {dtype:11s} {cs['n']:6d} {cs['triggered']:10d} {cs['trig_rate']:7.2f} {cs['gen']:6d} {cs['ver']:6d} {cs['ver_rate']:7.2f}")

    # Summary by type
    print("\nSummary by task type (E25):")
    for dtype, doms in [("DEPENDENT", DEPENDENT), ("independent", INDEPENDENT)]:
        total_gen = sum(codream_stats(e25_tasks, d)["gen"] for d in doms)
        total_ver = sum(codream_stats(e25_tasks, d)["ver"] for d in doms)
        total_trig = sum(codream_stats(e25_tasks, d)["triggered"] for d in doms)
        total_tasks = sum(codream_stats(e25_tasks, d)["n"] for d in doms)
        ver_rate = total_ver / total_gen if total_gen > 0 else 0
        trig_rate = total_trig / total_tasks if total_tasks > 0 else 0
        print(f"  {dtype:11s}: {trig_rate:.1%} tasks triggered, {total_gen} gen → {total_ver} ver ({ver_rate:.1%} pass)")

    # 4. Key interpretation
    print("\n--- 4. Key findings ---")
    for dom in ["math", "gsm8k", "mbpp", "humaneval"]:
        e17_q = quartile_stats(e17_tasks, dom)
        e25_q = quartile_stats(e25_tasks, dom)
        cs = codream_stats(e25_tasks, dom)
        if e17_q and e25_q:
            msg = "IMPROVED" if e25_q["mean"] > e17_q["mean"] + 0.005 else ("SIMILAR" if abs(e25_q["mean"] - e17_q["mean"]) <= 0.005 else "WORSE")
            print(f"  {dom:12s}: E25 mean={e25_q['mean']:.3f} vs E17={e17_q['mean']:.3f} → {msg}")
            print(f"             CoDream trigger rate={cs['trig_rate']:.1%}, verify={cs['ver_rate']:.1%}")
            print(f"             Q1 trend: E17={e17_q['trend']:+.3f} → E25={e25_q['trend']:+.3f}")

    print("\nPredicted outcome of E25:")
    print("  MATH: +0.03-0.05 (disagreement trigger catches ~42% more MATH tasks)")
    print("  GSM8K: +0.01-0.02 (domain_general strategies help arithmetic verification)")
    print("  MBPP/HE: similar or slight improvement (domain_general code strategies)")
    print("  Overall: should exceed E17's 0.874")
    print("=" * 90)


if __name__ == "__main__":
    main()
