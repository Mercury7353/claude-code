"""
EvoPool Selection Policy.

score(agent_i, task) = w1*affinity + w2*diversity_bonus + w3*collab + w4*exploration

- affinity: how well the agent's skill_memory matches the task type
- diversity_bonus: reward for adding domain diversity to the current team
- historical_collab_score: learned pair synergy from past collaborations
- exploration: UCB-style bonus that rewards under-selected agents; decays as
  an agent participates in more tasks, so early stages explore broadly and
  later stages exploit the core. Combined with random tie-breaking, this
  prevents the cold-start lock-in where a handful of agents are chosen
  deterministically on task 0 (all affinities tied at the default value)
  and then accumulate all experience via positive feedback.

Cold start: historical_collab_score = 0 for the first MIN_COLLAB_HISTORY tasks.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


# Weight defaults (can be tuned)
W_AFFINITY = 0.5
W_DIVERSITY = 0.3
W_COLLAB = 0.2
W_EXPLORE = 0.3  # UCB bonus weight; see _exploration_bonus for derivation

MIN_COLLAB_HISTORY = 5  # tasks before collab score kicks in


class CollabScoreTable:
    """Persistent pairwise collaboration score table.

    Also tracks per-agent participation counts used by the UCB-style
    exploration bonus in select_team.
    """

    def __init__(self):
        # (agent_id_a, agent_id_b) -> list of joint task scores
        self._scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._task_count: int = 0
        # agent_id -> number of tasks the agent has been on a team for
        self._agent_task_count: dict[str, int] = defaultdict(int)

    def record_team_result(self, team: list[Agent], task_score: float) -> None:
        """Record the outcome for all pairs in a team, and increment
        per-agent participation counts."""
        agent_ids = [a.agent_id for a in team]
        for aid in agent_ids:
            self._agent_task_count[aid] += 1
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                key = (agent_ids[i], agent_ids[j])
                self._scores[key].append(task_score)
        self._task_count += 1

    def get_pair_score(self, agent_id_a: str, agent_id_b: str) -> float:
        """Return mean historical score for a pair, or 0 if insufficient history."""
        if self._task_count < MIN_COLLAB_HISTORY:
            return 0.0
        key = (agent_id_a, agent_id_b)
        alt_key = (agent_id_b, agent_id_a)
        scores = self._scores.get(key, self._scores.get(alt_key, []))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_team_score(self, agent_ids: list[str]) -> float:
        """Return mean pair score for a team (mean over all pairs)."""
        pairs = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                pairs.append(self.get_pair_score(agent_ids[i], agent_ids[j]))
        return sum(pairs) / len(pairs) if pairs else 0.0

    def get_agent_count(self, agent_id: str) -> int:
        """Number of past tasks the agent has been selected for."""
        return self._agent_task_count.get(agent_id, 0)

    @property
    def task_count(self) -> int:
        return self._task_count

    def to_dict(self) -> dict:
        return {
            "scores": {f"{k[0]}|{k[1]}": v for k, v in self._scores.items()},
            "task_count": self._task_count,
            "agent_task_count": dict(self._agent_task_count),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CollabScoreTable:
        table = cls()
        table._task_count = d.get("task_count", 0)
        for key_str, scores in d.get("scores", {}).items():
            parts = key_str.split("|", 1)
            if len(parts) == 2:
                table._scores[(parts[0], parts[1])] = scores
        for aid, n in d.get("agent_task_count", {}).items():
            table._agent_task_count[aid] = n
        return table


def select_team(
    pool: list[Agent],
    task: dict,
    collab_table: CollabScoreTable,
    k: int = 3,
    w_affinity: float = W_AFFINITY,
    w_diversity: float = W_DIVERSITY,
    w_collab: float = W_COLLAB,
    w_explore: float = W_EXPLORE,
    rng: random.Random | None = None,
) -> list[Agent]:
    """
    Select k agents from the pool for a task using the EvoPool selection policy.

    Uses greedy sequential selection: add agents one by one, each time picking
    the agent that maximizes the incremental score. The score includes a
    UCB-style exploration bonus that favors under-selected agents and decays
    as agents participate; ties are broken randomly to prevent iteration-
    order lock-in at cold start.
    """
    if len(pool) <= k:
        return list(pool)

    if rng is None:
        rng = random.Random()

    task_type = task.get("type", "general")
    total_tasks = collab_table.task_count
    selected: list[Agent] = []
    remaining = list(pool)

    for _ in range(k):
        # Score every remaining candidate.
        scores: list[tuple[float, Agent]] = []
        for candidate in remaining:
            score = _compute_score(
                candidate=candidate,
                task_type=task_type,
                current_team=selected,
                collab_table=collab_table,
                total_tasks=total_tasks,
                w_affinity=w_affinity,
                w_diversity=w_diversity,
                w_collab=w_collab,
                w_explore=w_explore,
            )
            scores.append((score, candidate))

        # Pick the max score, breaking ties uniformly at random.
        # (Without random tie-breaking, cold-start scores are all equal and
        # the first agent in iteration order always wins, which is exactly
        # the lock-in we saw empirically.)
        max_score = max(s for s, _ in scores)
        top_candidates = [a for s, a in scores if s >= max_score - 1e-9]
        best_agent = rng.choice(top_candidates)

        selected.append(best_agent)
        remaining.remove(best_agent)

    return selected


def _compute_score(
    candidate: Agent,
    task_type: str,
    current_team: list[Agent],
    collab_table: CollabScoreTable,
    total_tasks: int,
    w_affinity: float,
    w_diversity: float,
    w_collab: float,
    w_explore: float,
) -> float:
    affinity = _affinity_score(candidate, task_type)
    diversity = _diversity_bonus(candidate, current_team, task_type)
    collab = _collab_score(candidate, current_team, collab_table)
    explore = _exploration_bonus(candidate, collab_table, total_tasks)
    return (
        w_affinity * affinity
        + w_diversity * diversity
        + w_collab * collab
        + w_explore * explore
    )


def _exploration_bonus(
    candidate: Agent,
    collab_table: CollabScoreTable,
    total_tasks: int,
) -> float:
    """UCB1-style bonus that rewards under-selected agents.

    bonus = sqrt(2 * log(total + 1) / (n_selected + 1))

    The scale is set so that a never-selected agent receives a bonus that
    can overcome the typical affinity gap (affinity 1.0 vs default 0.2,
    scaled by W_AFFINITY=0.5 = 0.4) at moderate stream lengths; this
    prevents the positive-feedback lock-in where three agents with
    slightly-higher affinity monopolize selection forever.

    Numerical examples with W_EXPLORE=0.3:
      total=10,  n=0  -> 0.3 * sqrt(2*2.4/1)   = 0.66
      total=10,  n=10 -> 0.3 * sqrt(2*2.4/11)  = 0.20
      total=100, n=0  -> 0.3 * sqrt(2*4.6/1)   = 0.91
      total=100, n=100-> 0.3 * sqrt(2*4.6/101) = 0.09

    Once the pool reaches saturation (every agent has nontrivial experience),
    affinity dominates and the bonus becomes a small perturbation.
    """
    n = collab_table.get_agent_count(candidate.agent_id)
    return math.sqrt(2.0 * math.log(total_tasks + 1.0) / (n + 1.0))


def _affinity_score(agent: Agent, task_type: str) -> float:
    """How well does the agent's skill match the task type?"""
    return agent.profile.skill_memory.get(task_type, 0.2)


def _diversity_bonus(
    candidate: Agent,
    current_team: list[Agent],
    task_type: str = "",
) -> float:
    """
    Reward for adding domain diversity to the current team.
    Higher = candidate covers domains not already covered by team.

    Key fix: penalize off-domain diversity. If the task has a known type
    and the current team already has strong task affinity, cross-domain
    diversity should NOT outweigh relevance.  We scale the diversity bonus
    by the candidate's affinity for the task type so that off-domain agents
    (high diversity but zero relevance) don't crowd out specialists.
    """
    if not current_team:
        return 0.5  # Neutral for first selection

    team_domains = {
        domain
        for agent in current_team
        for domain, score in agent.profile.skill_memory.items()
        if score > 0.5
    }
    candidate_domains = {
        domain
        for domain, score in candidate.profile.skill_memory.items()
        if score > 0.5
    }
    new_domains = candidate_domains - team_domains
    if not candidate_domains:
        raw_diversity = 0.0
    else:
        raw_diversity = len(new_domains) / len(candidate_domains)

    # Scale by task relevance: an off-domain agent's diversity bonus is
    # dampened in proportion to how irrelevant it is for this task.
    # Agents below the affinity floor get near-zero diversity credit, preventing
    # code specialists from joining math teams based purely on "diverse" skills.
    if task_type:
        task_affinity = candidate.profile.skill_memory.get(task_type, 0.2)
        # Hard gate: below 0.3 affinity, diversity contributes almost nothing.
        # This ensures domain specialists always outrank off-domain generalists.
        if task_affinity < 0.3:
            return raw_diversity * 0.1
        # Above threshold: linearly scale from 0.3→0.5 affinity
        relevance_scale = min(1.0, (task_affinity - 0.3) / 0.2)
        return raw_diversity * relevance_scale
    return raw_diversity


def _collab_score(
    candidate: Agent,
    current_team: list[Agent],
    collab_table: CollabScoreTable,
) -> float:
    """Mean historical collaboration score between candidate and each current team member."""
    if not current_team:
        return 0.0
    scores = [
        collab_table.get_pair_score(candidate.agent_id, member.agent_id)
        for member in current_team
    ]
    return sum(scores) / len(scores)
