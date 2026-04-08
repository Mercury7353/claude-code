#!/usr/bin/env python3
"""
Analyze within-domain learning curves for AIME experiments.
Generates per-domain bin accuracy tables for the difficulty ladder story.
"""

import json
import os
import sys

RESULTS_BASE = "/nfs/hpc/share/zhanyaol/claude-code/results"

# Conditions to compare (name, path, color-hint)
HARD_MATH_CONDITIONS = [
    ("Single-Agent (E39)", "e39/single_agent_hard_math_stream_seed42.json"),
    ("MemCollab (E52)", "e52/memcollab_hard_math_stream_seed42.json"),
    ("EvoMem (E53)", "e53/evomem_hard_math_stream_seed42.json"),
    ("AgentNet (E51)", "e51/agentnet_hard_math_stream_seed42.json"),
    ("DyLAN old (E42)", "e42/dylan_hard_math_stream_seed42.json"),
    ("DyLAN fixed (E54)", "e54/dylan_hard_math_stream_seed42.json"),
    ("noCoDream (E41)", "e41/evopool_no_codream_hard_math_stream_seed42.json"),
    ("noLifecycle (E48)", "e48/evopool_no_lifecycle_hard_math_stream_seed42.json"),
    ("noL2 (E49)", "e49/evopool_no_l2_hard_math_stream_seed42.json"),
    ("noVerify (E50)", "e50/evopool_no_verify_hard_math_stream_seed42.json"),
    ("EvoPool-full (E40)", "e40/evopool_full_hard_math_stream_seed42.json"),
]

DOMAINS = ["math_hard", "aime_2022", "aime_2023", "aime_2024", "aime_2025"]
DOMAIN_SIZES = {"math_hard": 30, "aime_2022": 30, "aime_2023": 30, "aime_2024": 30, "aime_2025": 15}


def load(path):
    full = os.path.join(RESULTS_BASE, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def extract_domain_tasks(data, domain):
    """Return list of (task_index_in_domain, score) for the given domain."""
    tasks = data.get("per_task_results", [])
    if not tasks:
        return []
    return [(i, t["score"]) for i, t in enumerate(t for t in tasks if t.get("domain") == domain)]


def bin_accuracy(domain_tasks, bin_size=10):
    """Split domain_tasks into bins of bin_size, return list of (bin_label, mean)."""
    bins = []
    for start in range(0, len(domain_tasks), bin_size):
        chunk = domain_tasks[start:start + bin_size]
        if not chunk:
            continue
        mean = sum(s for _, s in chunk) / len(chunk)
        label = f"{start+1}-{start+len(chunk)}"
        bins.append((label, mean, len(chunk)))
    return bins


def print_domain_curves(domain):
    """Print per-bin accuracy for all conditions in a domain."""
    print(f"\n{'='*80}")
    print(f"Within-Domain Learning Curve: {domain.upper()}")
    print(f"{'='*80}")

    results = {}
    for name, path in HARD_MATH_CONDITIONS:
        data = load(path)
        if data is None:
            continue
        dt = extract_domain_tasks(data, domain)
        if not dt:
            continue
        bins = bin_accuracy(dt, bin_size=10)
        results[name] = bins

    if not results:
        print("  No completed results for this domain yet.")
        return

    # Header
    max_bins = max(len(b) for b in results.values())
    header = f"  {'Condition':<28}"
    for i in range(max_bins):
        header += f"  B{i+1:4}"
    header += f"  {'Mean':>6}  {'Trend':>6}"
    print(header)
    print("  " + "-" * (28 + max_bins * 7 + 16))

    for name, bins in sorted(results.items(), key=lambda x: -sum(m for _, m, _ in x[1])):
        row = f"  {name:<28}"
        for label, mean, n in bins:
            row += f"  {mean:.3f}"
        # Pad if fewer bins
        for _ in range(max_bins - len(bins)):
            row += f"  {'—':>5}"
        overall = sum(m for _, m, _ in bins) / len(bins)
        trend = bins[-1][1] - bins[0][1] if len(bins) >= 2 else 0.0
        row += f"  {overall:6.3f}  {trend:+6.3f}"
        print(row)


def print_summary_table():
    """Print full summary table across all domains."""
    print("\n" + "="*100)
    print("Hard Math Stream — Full Results Table")
    print("="*100)

    header = f"{'Condition':<28}"
    for d in DOMAINS:
        short = d.replace("aime_", "A").replace("math_", "MH")[:6]
        header += f"  {short:>6}"
    header += f"  {'MEAN':>6}"
    print(header)
    print("-"*100)

    rows = []
    for name, path in HARD_MATH_CONDITIONS:
        data = load(path)
        if data is None:
            continue
        tasks = data.get("per_task_results", [])
        if not tasks:
            continue
        ds = {}
        for t in tasks:
            dom = t.get("domain", "unk")
            ds.setdefault(dom, []).append(t["score"])
        means = {d: sum(ds[d]) / len(ds[d]) for d in DOMAINS if d in ds}
        if not means:
            continue
        overall = sum(means.values()) / len(means)
        rows.append((name, means, overall))

    rows.sort(key=lambda x: -x[2])
    for name, means, overall in rows:
        row = f"{name:<28}"
        for d in DOMAINS:
            v = means.get(d)
            row += f"  {v:6.3f}" if v is not None else f"  {'—':>6}"
        row += f"  {overall:6.3f}"
        print(row)

    print("="*100)


def print_evopool_vs_single_delta():
    """Print EvoPool vs Single-Agent gain per domain and within-domain trend comparison."""
    e40 = load("e40/evopool_full_hard_math_stream_seed42.json")
    e39 = load("e39/single_agent_hard_math_stream_seed42.json")
    e41 = load("e41/evopool_no_codream_hard_math_stream_seed42.json")

    if not e39:
        print("E39 (single-agent) not found")
        return

    print("\n" + "="*80)
    print("KEY COMPARISON: EvoPool vs noCoDream vs Single (within-domain trends)")
    print("="*80)

    for domain in DOMAINS:
        print(f"\n--- {domain.upper()} ---")
        for name, data in [("Single (E39)", e39), ("noCoDream (E41)", e41), ("EvoPool (E40)", e40)]:
            if data is None:
                print(f"  {name}: NOT YET AVAILABLE")
                continue
            dt = extract_domain_tasks(data, domain)
            if not dt:
                print(f"  {name}: no tasks in this domain yet")
                continue
            bins = bin_accuracy(dt, bin_size=10)
            bin_str = "  ".join(f"B{i+1}={m:.3f}" for i, (_, m, _) in enumerate(bins))
            trend = bins[-1][1] - bins[0][1] if len(bins) >= 2 else 0.0
            overall = sum(m for _, m, _ in bins) / len(bins)
            print(f"  {name:<20}: {bin_str}  | mean={overall:.3f}  trend={trend:+.3f}")


if __name__ == "__main__":
    print_summary_table()
    print_evopool_vs_single_delta()
    for domain in DOMAINS:
        print_domain_curves(domain)
