#!/usr/bin/env python3
"""
Paper figures for EvoPool.

Generates:
  1. fig1_learning_curve.pdf  — smoothed running mean over 600 tasks (all conditions)
  2. fig2_domain_bar.pdf      — per-domain bar chart (main results table)
  3. fig3_codream_ablation.pdf — CoDream ablation: HQA/DROP before/after

Usage:
  cd /nfs/hpc/share/zhanyaol/claude-code
  python figures/plot_results.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ────────────────────────────────────────────────────────────────────
BASE = "/nfs/hpc/share/zhanyaol/claude-code/results"
OUT  = "/nfs/hpc/share/zhanyaol/claude-code/figures"

EXPERIMENTS = {
    "Single-Agent":         "e18/single_agent_aflow_stream_seed42.json",
    "SC k=5":               "e10/self_consistency_aflow_stream_seed42.json",
    "AFlow† (Qwen3-8B)":    "e9/aflow_aflow_stream_seed42.json",
    "DyLAN":                "e13/dylan_aflow_stream_seed42.json",
    "EvoPool -CoDream":     "e15/evopool_no_codream_aflow_stream_seed42.json",
    "EvoPool (ours)":       "e17/evopool_full_aflow_stream_seed42.json",
}

COLORS = {
    "Single-Agent":         "#9CA3AF",   # gray
    "SC k=5":               "#6B7280",   # dark gray
    "AFlow† (Qwen3-8B)":    "#F59E0B",   # amber
    "DyLAN":                "#3B82F6",   # blue
    "EvoPool -CoDream":     "#8B5CF6",   # purple
    "EvoPool (ours)":       "#10B981",   # green (hero)
}

LINESTYLES = {
    "Single-Agent":         ":",
    "SC k=5":               "--",
    "AFlow† (Qwen3-8B)":    "-.",
    "DyLAN":                "--",
    "EvoPool -CoDream":     "-",
    "EvoPool (ours)":       "-",
}

LINEWIDTHS = {
    "Single-Agent":         1.4,
    "SC k=5":               1.4,
    "AFlow† (Qwen3-8B)":    1.6,
    "DyLAN":                1.8,
    "EvoPool -CoDream":     2.0,
    "EvoPool (ours)":       2.8,
}

DOMAINS     = ["gsm8k", "hotpotqa", "mbpp", "math", "humaneval", "drop"]
DOMAIN_LABELS = ["GSM8K", "HotpotQA", "MBPP", "MATH", "HumanEval", "DROP"]
DOMAIN_TICKS  = [0, 100, 200, 300, 400, 500]   # task index where each domain starts


def load(name):
    path = os.path.join(BASE, EXPERIMENTS[name])
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def smooth(scores, w=20):
    """Running average with window w."""
    out = []
    for i in range(len(scores)):
        lo = max(0, i - w + 1)
        out.append(np.mean(scores[lo:i+1]))
    return out


# ── Figure 1: Learning Curve ─────────────────────────────────────────────────
def fig_learning_curve():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name in EXPERIMENTS:
        data = load(name)
        if data is None:
            print(f"  [skip] {name}: file not found")
            continue
        scores = data["all_scores"]
        xs = list(range(1, len(scores) + 1))
        ys = smooth(scores, w=20)
        ax.plot(xs, ys,
                label=name,
                color=COLORS[name],
                linestyle=LINESTYLES[name],
                linewidth=LINEWIDTHS[name],
                alpha=0.92)

    # domain boundary lines
    for tick, dlabel in zip(DOMAIN_TICKS[1:], DOMAIN_LABELS[1:]):
        ax.axvline(tick, color="#D1D5DB", lw=1, ls=":")
    # domain name annotations
    for i, (tick, dlabel) in enumerate(zip(DOMAIN_TICKS, DOMAIN_LABELS)):
        ax.text(tick + 5, 0.08, dlabel, fontsize=7.5, color="#6B7280",
                ha="left", va="bottom")

    ax.set_xlabel("Task Index", fontsize=11)
    ax.set_ylabel("Running Mean Score (w=20)", fontsize=11)
    ax.set_title("EvoPool vs. Baselines on AFlow-Stream Benchmark\n(Qwen3-8B, training-free)", fontsize=12)
    ax.set_xlim(1, 600)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(OUT, "fig1_learning_curve.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Figure 2: Per-Domain Bar Chart ───────────────────────────────────────────
def fig_domain_bar():
    names = list(EXPERIMENTS.keys())
    loaded = {n: load(n) for n in names}

    # collect domain scores
    domain_data = {n: {} for n in names}
    for name, data in loaded.items():
        if data is None:
            continue
        ds = data.get("domain_scores", {})
        for d in DOMAINS:
            scores = ds.get(d, [])
            domain_data[name][d] = np.mean(scores) if scores else None

    n_methods = len(names)
    n_domains  = len(DOMAINS)
    x = np.arange(n_domains)
    bar_w = 0.13
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_w

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, name in enumerate(names):
        vals = [domain_data[name].get(d) for d in DOMAINS]
        xs_valid = [x[j] + offsets[i] for j, v in enumerate(vals) if v is not None]
        ys_valid = [v for v in vals if v is not None]
        xs_all   = [x[j] + offsets[i] for j in range(n_domains)]
        ys_all   = [v if v is not None else 0 for v in vals]
        bars = ax.bar(xs_all, ys_all, bar_w * 0.92,
                      color=COLORS[name], alpha=0.85, label=name,
                      edgecolor="white", linewidth=0.5)
        # mark missing bars with hatching
        for j, v in enumerate(vals):
            if v is None:
                ax.bar(x[j] + offsets[i], 0.02, bar_w * 0.92,
                       color="none", edgecolor="#9CA3AF", linewidth=0.8,
                       hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_LABELS, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Per-Domain Performance (Qwen3-8B, 100 tasks/domain)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # mean score annotation above each group
    for i, name in enumerate(names):
        data = loaded[name]
        if data is None:
            continue
        mean = data["summary"].get("mean_score", 0)
        # annotate slightly above the tallest bar in this group
        max_v = max((domain_data[name].get(d) or 0) for d in DOMAINS)
        # skip per-bar label clutter; instead add to legend label (already in legend)

    out_path = os.path.join(OUT, "fig2_domain_bar.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Figure 3: CoDream Ablation ────────────────────────────────────────────────
def fig_codream_ablation():
    """Side-by-side comparison showing CoDream impact on HQA and DROP."""
    codream_off = load("EvoPool -CoDream")   # E15b
    codream_on  = load("EvoPool (ours)")     # E17

    if codream_off is None or codream_on is None:
        print("  [skip] CoDream ablation: E15b or E17 not found")
        return

    domains_focus = ["hotpotqa", "drop", "gsm8k", "math", "mbpp", "humaneval"]
    labels        = ["HotpotQA", "DROP", "GSM8K", "MATH", "MBPP", "HumanEval"]

    def get_mean(data, domain):
        scores = data["domain_scores"].get(domain, [])
        return np.mean(scores) if scores else 0.0

    vals_off = [get_mean(codream_off, d) for d in domains_focus]
    vals_on  = [get_mean(codream_on,  d) for d in domains_focus]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_off = ax.bar(x - w/2, vals_off, w, label="EvoPool -CoDream (E15b)",
                      color="#8B5CF6", alpha=0.85, edgecolor="white")
    bars_on  = ax.bar(x + w/2, vals_on,  w, label="EvoPool +CoDream (E17, ours)",
                      color="#10B981", alpha=0.85, edgecolor="white")

    # delta annotations for HQA and DROP
    for i in range(2):  # first two are HQA, DROP
        delta = vals_on[i] - vals_off[i]
        x_pos = x[i] + w/2
        y_pos = vals_on[i] + 0.02
        ax.text(x_pos, y_pos, f"+{delta:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#059669")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("CoDream Ablation: Impact on Multi-Hop Reasoning", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # shade HQA+DROP bars to highlight
    ax.axvspan(-0.5, 1.5, alpha=0.06, color="#059669")
    ax.text(0.5, 1.07, "multi-hop\nreasoning", ha="center", va="top",
            fontsize=8, color="#059669", style="italic")

    out_path = os.path.join(OUT, "fig3_codream_ablation.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Figure 4: CoDream insight distribution ───────────────────────────────────
def fig_insight_distribution():
    """Pie chart of L1/L2/L3 insight distribution from E17."""
    data = load("EvoPool (ours)")
    if data is None:
        print("  [skip] Insight distribution: E17 not found")
        return

    # Try to get from metrics_log
    metrics = data.get("metrics_log", [])
    l_counts = {"L1": 0, "L2": 0, "L3": 0}
    for m in metrics:
        for k in l_counts:
            l_counts[k] += m.get(f"insight_{k.lower()}_count", 0)

    total = sum(l_counts.values())
    if total == 0:
        print("  [skip] No insight distribution data in E17 metrics_log")
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = [f"L1 Working\n(task-specific)", f"L2 Subdomain\n(domain-scoped)", f"L3 General\n(cross-domain)"]
    sizes  = [l_counts["L1"], l_counts["L2"], l_counts["L3"]]
    colors = ["#DBEAFE", "#A7F3D0", "#FDE68A"]
    explode = [0, 0.05, 0.1]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title("CoDream Insight Distribution\nby Transferability Level (E17)", fontsize=11)

    out_path = os.path.join(OUT, "fig4_insight_distribution.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Figure 5: Within-Domain Learning (CoDream vs noCoDream) ─────────────────
def fig_within_domain_learning():
    """
    Shows Q1/Q4 (first 25 / last 25 tasks) within HQA and DROP for CoDream vs noCoDream.
    Demonstrates: CoDream enables learning; noCoDream shows stagnation/decline.
    Also shows the initial gap (Q1) as evidence of cross-domain L3 transfer.
    """
    codream_off = load("EvoPool -CoDream")   # E15b (no CoDream)
    # Use E17 if available (clean code), fall back to E12 (old code but CoDream active)
    codream_on  = load("EvoPool (ours)")
    e12_label   = "EvoPool +CoDream (E17)"
    if codream_on is None:
        import os as _os
        _p = _os.path.join(BASE, "e12/evopool_full_aflow_stream_seed42.json")
        if _os.path.exists(_p):
            import json as _json
            with open(_p) as _f:
                codream_on = _json.load(_f)
            e12_label = "EvoPool +CoDream (E12)"
    if codream_off is None or codream_on is None:
        print("  [skip] Within-domain learning: E12 or E15b not found")
        return

    def quartile_means(data, domain, n_q=4):
        scores = data["domain_scores"].get(domain, [])
        q = len(scores) // n_q
        return [np.mean(scores[i*q:(i+1)*q]) for i in range(n_q)] if q > 0 else []

    domains_focus = ["hotpotqa", "drop"]
    d_labels      = ["HotpotQA", "DROP"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, domain, dlabel in zip(axes, domains_focus, d_labels):
        q_off = quartile_means(codream_off, domain)
        q_on  = quartile_means(codream_on,  domain)
        xs = [1, 2, 3, 4]
        ax.plot(xs, q_on,  "o-", color="#10B981", lw=2.5, ms=7, label=e12_label)
        ax.plot(xs, q_off, "s--", color="#8B5CF6", lw=2, ms=6, label="EvoPool -CoDream (E15b)")
        ax.fill_between(xs, q_off, q_on, alpha=0.12, color="#10B981")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Q1\n(first 25)", "Q2", "Q3", "Q4\n(last 25)"], fontsize=9)
        ax.set_ylabel("Accuracy" if ax == axes[0] else "", fontsize=11)
        ax.set_title(f"{dlabel}: Learning Dynamics", fontsize=11)
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # annotate Q1 gap (cross-domain transfer)
        if q_on and q_off:
            gap = q_on[0] - q_off[0]
            ax.annotate(f"Q1 gap\n(L3 transfer)\n+{gap:.2f}", xy=(1, q_on[0]),
                        xytext=(1.4, q_on[0]+0.05),
                        fontsize=7.5, color="#059669",
                        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2))

    fig.suptitle("CoDream Enables Learning; Without CoDream, Performance Stagnates",
                 fontsize=12, y=1.02)

    out_path = os.path.join(OUT, "fig5_within_domain_learning.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Figure 6: Task Dependence Concept ─────────────────────────────────────────
def fig_task_dependence():
    """
    Spider/radar chart showing per-domain gain from CoDream (E17 vs E15b) or (E12 vs E15b).
    Illustrates that CoDream benefit correlates with task dependence (multi-hop > independent).
    """
    codream_off = load("EvoPool -CoDream")    # E15b
    codream_on  = load("EvoPool (ours)")      # E17 (preferred)
    if codream_on is None:
        import os as _os, json as _json
        _p = _os.path.join(BASE, "e12/evopool_full_aflow_stream_seed42.json")
        if _os.path.exists(_p):
            with open(_p) as _f:
                codream_on = _json.load(_f)
    if codream_off is None or codream_on is None:
        print("  [skip] Task dependence: data not available")
        return

    # Per-domain CoDream gain
    def get_mean(data, d):
        s = data["domain_scores"].get(d, [])
        return np.mean(s) if s else 0.0

    gains = {}
    for d, dl in zip(DOMAINS, DOMAIN_LABELS):
        gains[dl] = get_mean(codream_on, d) - get_mean(codream_off, d)

    # "Task dependence" classification: high=multi-hop, low=independent
    dependence = {
        "GSM8K": "Independent\n(arithmetic)",
        "HotpotQA": "Dependent\n(multi-hop)",
        "MBPP": "Independent\n(code)",
        "MATH": "Semi-dep.\n(strategy)",
        "HumanEval": "Independent\n(code)",
        "DROP": "Dependent\n(multi-hop)",
    }
    dep_colors = {
        "Independent\n(arithmetic)": "#6B7280",
        "Independent\n(code)": "#6B7280",
        "Semi-dep.\n(strategy)": "#F59E0B",
        "Dependent\n(multi-hop)": "#10B981",
    }

    domains_ordered = ["HotpotQA", "DROP", "MATH", "GSM8K", "MBPP", "HumanEval"]
    gains_ordered   = [gains[d] for d in domains_ordered]
    colors_ordered  = [dep_colors[dependence[d]] for d in domains_ordered]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(domains_ordered))
    bars = ax.bar(xs, gains_ordered, color=colors_ordered, alpha=0.85, edgecolor="white", lw=0.5)

    # y=0 line
    ax.axhline(0, color="#9CA3AF", lw=1)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\n({dependence[d].split(chr(10))[0]})" for d in domains_ordered],
                        fontsize=9)
    ax.set_ylabel("CoDream Gain (accuracy)", fontsize=11)
    ax.set_title("Co-Dream Benefit Aligns with Task Dependence\n(Gains largest for multi-hop reasoning tasks)",
                 fontsize=11)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#10B981", label="Dependent (multi-hop reasoning)"),
        mpatches.Patch(color="#F59E0B", label="Semi-dependent (strategy)"),
        mpatches.Patch(color="#6B7280", label="Independent (each task self-contained)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(OUT, "fig6_task_dependence.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating paper figures...")
    fig_learning_curve()
    fig_domain_bar()
    fig_codream_ablation()
    fig_insight_distribution()
    fig_within_domain_learning()
    fig_task_dependence()
    print("Done.")
