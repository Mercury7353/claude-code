"""
EvoPool contextual quality-diversity selector (anchor + complement + scout).

Replaces the earlier greedy scalar-score selector, which cold-started by
iterating over agents and deterministically picking the first one in a
tie (yielding a 3-agent lock-in), and the subsequent UCB1 band-aid which
converges to a few global best arms — the wrong objective for a system
whose motivation is emergent *niche-based* specialization.

Design (gpt-5.4-pro consult, 2026-04-12, with user-driven modification):

  Anchor     = argmax_i  q_i(z_t)                                    (random tiebreak)
  Complement = argmax_j  q_j(z_t) + gamma * syn(anchor,j,z_t)
                                  - mu    * overlap(anchor,j)        (j != anchor)
  Scout      = argmax_k  u_k(z_t) + rho   * coverage_gap(k,z_t)
                                  - mu'   * overlap(k, {anchor,comp})   (k not in prev)

Why this produces niche specialization by construction:
- q_i(z) is *per niche*, so a generalist can no longer win every task with a
  single high scalar affinity. Each niche has its own champion.
- Anchor purely exploits best q on the current niche — no crowding penalty
  (per user: if A is genuinely the geometry champion, letting A always win
  geometry IS specialization; the lock-in we were trying to fix was global,
  not per-niche).
- Complement picks a second agent that is competent on the niche but not
  redundant with the anchor (low style overlap, positive pair synergy) —
  opens bridge/orthogonal agents even when they are not the niche champion.
- Scout gives under-exposed agents a seat in the niche's neighborhood so
  that new agents, forked clones, and long-tail pool members can accumulate
  niche experience rather than being stuck at the default forever.

Cold start (all q_i(z)=0): anchor's argmax resolves via random tie-break;
different early tasks pick different anchors; q_i(z) diverges across agents;
specialists emerge. Uncertainty u_i(z) is high for all agents early, so the
scout position naturally drives exploration broadly at first.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


# Top-level weights
W_COMP_SYN = 0.3       # gamma: bonus for niche-local synergy with anchor
W_COMP_OVERLAP = 0.5   # mu:    penalty for style/memory overlap with anchor
W_SCOUT_COVGAP = 0.3   # rho:   bonus for filling uncovered niche regions
W_SCOUT_OVERLAP = 0.5  # mu':   penalty for overlap with anchor+complement

# Uncertainty model: u_i(z) = 1 / sqrt(1 + niche_n_i(z)); agents with no niche
# samples start at u=1.0, those with many samples fall toward 0. Simple,
# monotone, and robust.
MIN_COLLAB_HISTORY = 5


# =====================================================================
# CollabScoreTable — niche-conditional pair synergy tracking
# =====================================================================

class CollabScoreTable:
    """Pairwise collaboration score table, niche-conditioned.

    Records for each unordered pair (a, b) and niche z, the list of joint
    task rewards on past niche-z tasks. Global (niche-agnostic) queries
    still work for legacy callers.
    """

    def __init__(self):
        # (niche, agent_a, agent_b) -> list of joint task scores
        self._scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        # global pair history for legacy callers
        self._global_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._task_count: int = 0

    def record_team_result(self, team: list[Agent], task_score: float, niche: str = "") -> None:
        """Record the outcome for all pairs in a team. Stores both niche-
        conditional and global aggregates."""
        agent_ids = [a.agent_id for a in team]
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a, b = sorted([agent_ids[i], agent_ids[j]])
                if niche:
                    self._scores[(niche, a, b)].append(task_score)
                self._global_scores[(a, b)].append(task_score)
        self._task_count += 1

    def get_pair_score(self, agent_id_a: str, agent_id_b: str, niche: str = "") -> float:
        """Mean historical joint reward for a pair, optionally within a niche.
        Returns 0 before MIN_COLLAB_HISTORY total tasks are logged."""
        if self._task_count < MIN_COLLAB_HISTORY:
            return 0.0
        a, b = sorted([agent_id_a, agent_id_b])
        if niche:
            scores = self._scores.get((niche, a, b), [])
        else:
            scores = self._global_scores.get((a, b), [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_team_score(self, agent_ids: list[str], niche: str = "") -> float:
        """Mean pair score for a team (mean over all pairs)."""
        pairs = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                pairs.append(self.get_pair_score(agent_ids[i], agent_ids[j], niche))
        return sum(pairs) / len(pairs) if pairs else 0.0

    @property
    def task_count(self) -> int:
        return self._task_count

    def to_dict(self) -> dict:
        return {
            "scores": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self._scores.items()},
            "global_scores": {f"{k[0]}|{k[1]}": v for k, v in self._global_scores.items()},
            "task_count": self._task_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CollabScoreTable:
        table = cls()
        table._task_count = d.get("task_count", 0)
        for key_str, scores in d.get("scores", {}).items():
            parts = key_str.split("|", 2)
            if len(parts) == 3:
                table._scores[(parts[0], parts[1], parts[2])] = scores
        # Back-compat: old format stored 2-tuple keys in "scores" without niche.
        # Merge those into _global_scores.
        for key_str, scores in d.get("global_scores", {}).items():
            parts = key_str.split("|", 1)
            if len(parts) == 2:
                table._global_scores[(parts[0], parts[1])] = scores
        return table


# =====================================================================
# Per-agent helpers
# =====================================================================

def _niche_q(agent: Agent, niche: str) -> float:
    """EWMA of reward on tasks in `niche`. Default 0 (no prior)."""
    return agent.profile.niche_q.get(niche, 0.0)


def _niche_uncertainty(agent: Agent, niche: str) -> float:
    """1 / sqrt(1 + niche_n). High when agent has little experience on niche."""
    n = agent.profile.niche_n.get(niche, 0)
    return 1.0 / math.sqrt(1.0 + n)


def _style_signature(agent: Agent) -> list[float]:
    """Cheap style signature = sorted skill_memory values over a stable key
    order. Two agents with similar competence-over-domains vectors count as
    stylistically overlapping. Returns empty list if no skill memory yet."""
    if not agent.profile.skill_memory:
        return []
    keys = sorted(agent.profile.skill_memory.keys())
    return [agent.profile.skill_memory.get(k, 0.0) for k in keys]


def _overlap(a: Agent, b: Agent) -> float:
    """Cosine similarity of style signatures. Agents without overlapping
    skill-memory keys get overlap 0 (which correctly reads as "no overlap
    evidence yet", not "identical")."""
    # Build aligned vectors over union of keys
    ka = set(a.profile.skill_memory.keys())
    kb = set(b.profile.skill_memory.keys())
    keys = sorted(ka | kb)
    if not keys:
        return 0.0
    va = [a.profile.skill_memory.get(k, 0.0) for k in keys]
    vb = [b.profile.skill_memory.get(k, 0.0) for k in keys]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def _set_overlap(k: Agent, members: list[Agent]) -> float:
    if not members:
        return 0.0
    return max(_overlap(k, m) for m in members)


def _coverage_gap(agent: Agent, niche: str) -> float:
    """How under-represented this agent's *niche profile* is on the current
    niche. Cheap version: 1 - fraction of the agent's total niche exposures
    spent on this particular niche. Encourages diversifying an agent's niche
    footprint, not just piling more tasks onto an already-saturated niche."""
    total = sum(agent.profile.niche_n.values())
    if total == 0:
        return 1.0
    return 1.0 - (agent.profile.niche_n.get(niche, 0) / total)


# =====================================================================
# Team assembly
# =====================================================================

def select_team(
    pool: list[Agent],
    task: dict,
    collab_table: CollabScoreTable,
    k: int = 3,
    w_comp_syn: float = W_COMP_SYN,
    w_comp_overlap: float = W_COMP_OVERLAP,
    w_scout_covgap: float = W_SCOUT_COVGAP,
    w_scout_overlap: float = W_SCOUT_OVERLAP,
    rng: random.Random | None = None,
    **_legacy,  # absorb old `w_affinity`, `w_diversity`, `w_collab` kwargs silently
) -> list[Agent]:
    """Select k=3 agents via anchor + complement + scout roles, conditioned on
    the task's niche. See module docstring for full rationale.

    Larger k > 3 falls back to repeated scout picks after anchor+complement.
    """
    if len(pool) <= k:
        return list(pool)

    if rng is None:
        rng = random.Random()

    # Niche inference — identical to agent.get_niche() to keep state/selector
    # in sync without circular import. Priority: subtype > domain > type.
    niche = ""
    for key in ("subtype", "domain", "type"):
        v = task.get(key)
        if v:
            niche = str(v)
            break

    # ---- Anchor: argmax q_i(z_t), random tie-break ----
    anchor_scores = [(_niche_q(a, niche), a) for a in pool]
    max_a = max(s for s, _ in anchor_scores)
    anchor_candidates = [a for s, a in anchor_scores if s >= max_a - 1e-9]
    anchor = rng.choice(anchor_candidates)

    if k == 1:
        return [anchor]

    # ---- Complement: competence + synergy - overlap (j != anchor) ----
    comp_scores: list[tuple[float, Agent]] = []
    for a in pool:
        if a is anchor:
            continue
        q = _niche_q(a, niche)
        syn = collab_table.get_pair_score(anchor.agent_id, a.agent_id, niche)
        ovl = _overlap(anchor, a)
        score = q + w_comp_syn * syn - w_comp_overlap * ovl
        comp_scores.append((score, a))
    max_c = max(s for s, _ in comp_scores)
    comp_candidates = [a for s, a in comp_scores if s >= max_c - 1e-9]
    complement = rng.choice(comp_candidates)

    if k == 2:
        return [anchor, complement]

    # ---- Scout: uncertainty + coverage_gap - overlap(prev) ----
    chosen = [anchor, complement]
    remaining = [a for a in pool if a is not anchor and a is not complement]

    scout_scores: list[tuple[float, Agent]] = []
    for a in remaining:
        u = _niche_uncertainty(a, niche)
        cg = _coverage_gap(a, niche)
        ovl_prev = _set_overlap(a, chosen)
        score = u + w_scout_covgap * cg - w_scout_overlap * ovl_prev
        scout_scores.append((score, a))
    max_s = max(s for s, _ in scout_scores)
    scout_candidates = [a for s, a in scout_scores if s >= max_s - 1e-9]
    scout = rng.choice(scout_candidates)
    chosen.append(scout)

    # If k > 3, keep adding scouts (rare in practice).
    remaining = [a for a in remaining if a is not scout]
    while len(chosen) < k and remaining:
        more_scores: list[tuple[float, Agent]] = []
        for a in remaining:
            u = _niche_uncertainty(a, niche)
            cg = _coverage_gap(a, niche)
            ovl_prev = _set_overlap(a, chosen)
            more_scores.append((u + w_scout_covgap * cg - w_scout_overlap * ovl_prev, a))
        max_m = max(s for s, _ in more_scores)
        more_candidates = [a for s, a in more_scores if s >= max_m - 1e-9]
        pick = rng.choice(more_candidates)
        chosen.append(pick)
        remaining = [a for a in remaining if a is not pick]

    return chosen
