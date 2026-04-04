"""
Lightweight semantic task embedder for oracle-free domain detection.

Replaces hardcoded domain-cluster maps with continuous task-distance computation.
Uses TF-IDF bag-of-words embeddings so there is NO dependency on external
embedding models or API calls — works fully offline on the H100 node.

Key use cases:
  1. Fork cooldown: suppress fork if current task is semantically far from
     the agent's recent task history (we just domain-shifted).
  2. Co-Dream domain constraint: only apply skill updates when the updating
     insight's task is semantically close to the current task.
  3. Team selection: boost agents whose recent success episodes are
     semantically close to the current task.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


# ---------------------------------------------------------------------------
# Stopwords (very minimal — don't over-strip domain keywords)
# ---------------------------------------------------------------------------
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "for", "with", "at", "by", "from", "as", "this", "that", "it", "its",
    "and", "or", "not", "but", "if", "so",
})

# Domain-discriminative keywords given extra weight
_DOMAIN_KEYWORDS = {
    "python", "code", "function", "def", "return", "assert", "program",
    "class", "method", "algorithm", "implement", "math", "compute",
    "equation", "proof", "solve", "calculate", "value", "sum", "product",
    "passage", "text", "article", "answer", "question", "who", "what",
    "when", "where", "find", "list", "write", "generate", "given", "output",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP]


def embed(text: str) -> dict[str, float]:
    """
    Return a TF-IDF-style sparse embedding as a dict {term: weight}.
    Domain keywords get 2× weight to improve domain discrimination.
    """
    tokens = _tokenize(text[:500])  # cap to first 500 chars for speed
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values())
    vec: dict[str, float] = {}
    for term, cnt in counts.items():
        tf = cnt / total
        # Minimal IDF: domain keywords get boost
        idf = 2.0 if term in _DOMAIN_KEYWORDS else 1.0
        vec[term] = tf * idf
    # L2-normalise
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse embedding dicts."""
    if not a or not b:
        return 0.0
    # dot product over shared keys
    dot = sum(a[k] * b[k] for k in a if k in b)
    return dot  # already L2-normalised so |a|=|b|=1


def task_distance(task_a: dict, task_b: dict) -> float:
    """
    Semantic distance between two tasks (0=identical, 1=orthogonal).
    Uses the task's prompt text as the embedding source.
    """
    text_a = task_a.get("prompt", "") + " " + task_a.get("type", "")
    text_b = task_b.get("prompt", "") + " " + task_b.get("type", "")
    sim = cosine_sim(embed(text_a), embed(text_b))
    return 1.0 - sim


def is_domain_shift(
    current_task: dict,
    recent_tasks: Sequence[dict],
    threshold: float = 0.70,
    min_recent: int = 3,
) -> bool:
    """
    Return True if current_task is semantically far from recent_tasks,
    indicating a domain shift.

    threshold: minimum average similarity to recent tasks to be "same domain"
    If avg_sim < (1 - threshold), we declare a domain shift.
    """
    if len(recent_tasks) < min_recent:
        return False  # Not enough history to judge

    cur_emb = embed(current_task.get("prompt", "") + " " + current_task.get("type", ""))
    sims = []
    for t in recent_tasks[-min_recent:]:
        t_emb = embed(t.get("prompt", "") + " " + t.get("type", ""))
        sims.append(cosine_sim(cur_emb, t_emb))

    avg_sim = sum(sims) / len(sims)
    return avg_sim < (1.0 - threshold)


def tasks_related(task_a: dict, task_b: dict, threshold: float = 0.25) -> bool:
    """
    Return True if two tasks are semantically related (same domain cluster).
    Replaces the hardcoded _DOMAIN_CLUSTERS dict.
    threshold: cosine similarity above which tasks are considered related.
    """
    text_a = task_a.get("prompt", "") + " " + task_a.get("type", "")
    text_b = task_b.get("prompt", "") + " " + task_b.get("type", "")
    sim = cosine_sim(embed(text_a), embed(text_b))
    return sim >= threshold
