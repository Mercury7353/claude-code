"""
Lifelong learning evaluation metrics for EvoPool.

Key metrics from SWE-Bench-CL evaluation protocol:
- Forward Transfer (FWT): does past experience help future tasks?
- Backward Transfer (BWT): does new learning hurt old-domain performance?
- Lifelong performance curve: rolling mean score vs. task index
- Profile diversity: mean pairwise cosine distance of agent profiles
- Skill entropy: entropy of pool-wide skill distribution
"""

from __future__ import annotations

import math
from typing import Any


def compute_forward_transfer(
    scores_with_history: list[float],
    scores_no_history: list[float],
) -> float:
    """
    FWT = mean(score_t_with_history - score_t_no_history) for all t.
    Positive FWT means past experience helps future tasks.
    """
    if len(scores_with_history) != len(scores_no_history):
        n = min(len(scores_with_history), len(scores_no_history))
        scores_with_history = scores_with_history[:n]
        scores_no_history = scores_no_history[:n]
    diffs = [h - nh for h, nh in zip(scores_with_history, scores_no_history)]
    return sum(diffs) / len(diffs) if diffs else 0.0


def compute_backward_transfer(
    domain_scores_over_time: dict[str, list[float]],
) -> float:
    """
    BWT = mean over domains of (final_score_in_domain - score_right_after_domain_tasks).
    Negative BWT means forgetting; positive means continuous improvement.

    Args:
        domain_scores_over_time: domain -> list of scores over entire task stream
    """
    bwt_values = []
    for domain, scores in domain_scores_over_time.items():
        if len(scores) < 2:
            continue
        # Score right after the domain tasks (first 5 scores)
        initial_score = sum(scores[:5]) / len(scores[:5])
        # Final score in domain (last 5 scores)
        final_score = sum(scores[-5:]) / len(scores[-5:])
        bwt_values.append(final_score - initial_score)
    return sum(bwt_values) / len(bwt_values) if bwt_values else 0.0


def compute_rolling_mean(scores: list[float], window: int = 10) -> list[float]:
    """Compute rolling mean of scores with given window size."""
    result = []
    for i in range(len(scores)):
        window_start = max(0, i - window + 1)
        window_scores = scores[window_start:i + 1]
        result.append(sum(window_scores) / len(window_scores))
    return result


def compute_area_under_curve(rolling_means: list[float]) -> float:
    """
    AUC of the lifelong performance curve.
    Higher = better overall lifelong performance.
    """
    if not rolling_means:
        return 0.0
    return sum(rolling_means) / len(rolling_means)


def compute_learning_slope(rolling_means: list[float], last_n: int = 20) -> float:
    """
    Slope of the performance curve over the last N tasks.
    Positive = still improving; negative = forgetting/declining.
    """
    if len(rolling_means) < 2:
        return 0.0
    recent = rolling_means[-last_n:]
    n = len(recent)
    x_mean = (n - 1) / 2
    y_mean = sum(recent) / n
    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_results(
    system_name: str,
    scores: list[float],
    domain_scores: dict[str, list[float]] | None = None,
    baseline_scores: list[float] | None = None,
) -> dict:
    """
    Compute all metrics and return a summary dict.
    """
    rolling = compute_rolling_mean(scores)
    auc = compute_area_under_curve(rolling)
    slope = compute_learning_slope(rolling)
    bwt = compute_backward_transfer(domain_scores) if domain_scores else None
    fwt = compute_forward_transfer(scores, baseline_scores) if baseline_scores else None

    return {
        "system": system_name,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "final_score": sum(scores[-10:]) / 10 if len(scores) >= 10 else sum(scores) / max(len(scores), 1),
        "auc": auc,
        "learning_slope": slope,
        "bwt": bwt,
        "fwt": fwt,
        "n_tasks": len(scores),
    }


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    headers = ["System", "Mean Score", "Final Score", "AUC", "Slope", "BWT", "FWT"]
    rows = []
    for r in results:
        rows.append([
            r["system"],
            f"{r['mean_score']:.3f}",
            f"{r['final_score']:.3f}",
            f"{r['auc']:.3f}",
            f"{r['learning_slope']:.4f}",
            f"{r['bwt']:.3f}" if r["bwt"] is not None else "N/A",
            f"{r['fwt']:.3f}" if r["fwt"] is not None else "N/A",
        ])

    col_widths = [max(len(h), max(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * sum(col_widths + [3 * len(col_widths)]))
    for row in rows:
        print(fmt.format(*row))
