"""
Pool Lifecycle Operators for EvoPool.

Operators:
  - Specialize: reinforce domain strength for consistently strong agents
  - Fork: split agent handling divergent task types
  - Merge: consolidate two highly similar agents
  - Prune: remove consistently underperforming agents
  - Genesis: create new agent when pool lacks coverage for emerging task type

All operators require SUSTAINED patterns (>=3 task signals) to fire,
preventing instability from single-task noise.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, AgentProfile


# ------------------------------------------------------------------
# Lifecycle event log (for analysis / visualization)
# ------------------------------------------------------------------

@dataclass
class LifecycleEvent:
    event_type: str  # "specialize" | "fork" | "merge" | "prune" | "genesis"
    task_index: int
    agent_ids: list[str]
    new_agent_ids: list[str] = field(default_factory=list)
    reason: str = ""


# ------------------------------------------------------------------
# Trigger constants
# ------------------------------------------------------------------

SPECIALIZE_TRIGGER_TASKS = 5       # Need N tasks in domain before specializing
SPECIALIZE_MIN_SCORE = 0.75        # Mean score threshold for specialization
FORK_DIVERGENCE_TASKS = 8          # Need N tasks before considering fork (raised: more stable signal)
FORK_DIVERGENCE_THRESHOLD = 0.55   # Task type distribution entropy threshold (raised: less eager forking)
FORK_DOMAIN_COOLDOWN = 3           # Suppress fork if current task type only appeared in last N tasks
MERGE_SIMILARITY_THRESHOLD = 0.90  # Profile cosine similarity for merge
MERGE_MIN_TASKS = 10               # Both agents need this many tasks
PRUNE_UNDERPERFORMANCE_TASKS = 5   # Consecutive underperformance tasks
PRUNE_UNDERPERFORMANCE_RATIO = 0.8 # Agent score < this * pool_mean
GENESIS_COVERAGE_THRESHOLD = 0.4   # If no agent has affinity > this, trigger genesis


# ------------------------------------------------------------------
# Operator implementations
# ------------------------------------------------------------------

def check_and_apply_lifecycle(
    pool: list[Agent],
    task_index: int,
    recent_task_types: list[str],
    pool_mean_score: float,
    backbone_llm: str,
    n_min: int = 5,
    n_max: int = 50,
    current_task_type: str = "",
) -> tuple[list[Agent], list[LifecycleEvent]]:
    """
    Check all lifecycle conditions and apply operators as needed.
    Returns the updated pool and a list of lifecycle events that fired.
    """
    events: list[LifecycleEvent] = []
    pool = list(pool)  # shallow copy

    # Specialize first (no pool size change)
    for agent in pool:
        event = try_specialize(agent, task_index)
        if event:
            events.append(event)

    # Prune (reduces pool size — must be before fork/genesis)
    if len(pool) > n_min:
        pruned = []
        for agent in pool:
            event = try_prune(agent, pool_mean_score, task_index)
            if event:
                events.append(event)
                pruned.append(agent.agent_id)
        pool = [a for a in pool if a.agent_id not in pruned]

    # Merge (reduces pool size)
    if len(pool) > n_min:
        pool, merge_events = try_merge_pool(pool, task_index, backbone_llm)
        events.extend(merge_events)

    # Fork (increases pool size) — snapshot pool first to avoid mutation-during-iteration
    if len(pool) < n_max:
        forked_agents = []
        forked_parent_ids: set[str] = set()
        for agent in list(pool):  # iterate over snapshot
            if len(pool) - len(forked_parent_ids) + len(forked_agents) >= n_max:
                break
            new_agents, event = try_fork(agent, task_index, backbone_llm, current_task_type)
            if event:
                events.append(event)
                forked_agents.extend(new_agents)
                forked_parent_ids.add(agent.agent_id)
        pool = [a for a in pool if a.agent_id not in forked_parent_ids]
        pool.extend(forked_agents)

    # Genesis (increases pool size)
    if len(pool) < n_max and recent_task_types:
        new_agents, gen_events = try_genesis(pool, recent_task_types, task_index, backbone_llm)
        events.extend(gen_events)
        pool.extend(new_agents)

    return pool, events


def try_specialize(agent: Agent, task_index: int) -> LifecycleEvent | None:
    """Reinforce an agent's strongest domain if sustained performance warrants it."""
    for domain, scores in agent.profile.perf_stats.items():
        recent = scores[-SPECIALIZE_TRIGGER_TASKS:]
        if len(recent) < SPECIALIZE_TRIGGER_TASKS:
            continue
        if sum(recent) / len(recent) >= SPECIALIZE_MIN_SCORE:
            # Reinforce this domain
            old = agent.profile.skill_memory.get(domain, 0.5)
            agent.profile.skill_memory[domain] = min(1.0, old + 0.05)
            return LifecycleEvent(
                event_type="specialize",
                task_index=task_index,
                agent_ids=[agent.agent_id],
                reason=f"Sustained high performance in {domain} (mean={sum(recent)/len(recent):.2f})",
            )
    return None


def try_fork(
    agent: Agent,
    task_index: int,
    backbone_llm: str,
    current_task_type: str = "",
) -> tuple[list[Agent], LifecycleEvent | None]:
    """
    Fork an agent if its task history is sufficiently divergent.
    Creates two child agents with diverged profiles.

    Domain-boundary guard: if the current task type only appeared very recently
    (i.e. we just entered a new domain), suppress fork. Forking at domain
    boundaries replaces experienced agents with untrained children right when
    domain expertise is most needed (causing catastrophic forgetting).
    """
    from .agent import Agent as AgentClass, AgentProfile

    task_types = [t["type"] for t in agent.profile.task_history[-FORK_DIVERGENCE_TASKS:]]
    if len(task_types) < FORK_DIVERGENCE_TASKS:
        return [], None

    # Domain-boundary guard: if current task type appeared only in the last
    # FORK_DOMAIN_COOLDOWN tasks, we just transitioned domains — suppress fork
    if current_task_type:
        recent_types = task_types[-FORK_DOMAIN_COOLDOWN:]
        older_types = task_types[:-FORK_DOMAIN_COOLDOWN]
        if current_task_type in recent_types and current_task_type not in older_types:
            return [], None  # suppress: current domain is brand-new, don't disrupt

    # Compute distribution entropy over recent task types
    type_counts: dict[str, int] = {}
    for t in task_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    entropy = _entropy(list(type_counts.values()))
    if entropy < FORK_DIVERGENCE_THRESHOLD:
        return [], None

    # Get the two dominant types
    top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:2]
    if len(top_types) < 2:
        return [], None

    type_a, type_b = top_types[0][0], top_types[1][0]

    # Additional guard: if neither fork type matches the current task type,
    # the divergence is based on stale history — suppress to avoid disrupting
    # current-domain performance
    if current_task_type and current_task_type not in (type_a, type_b):
        return [], None

    # Create two child profiles
    child_a = AgentClass(
        profile=_fork_profile(agent.profile, dominant_type=type_a, backbone_llm=backbone_llm),
        parent_ids=[agent.agent_id],
    )
    child_b = AgentClass(
        profile=_fork_profile(agent.profile, dominant_type=type_b, backbone_llm=backbone_llm),
        parent_ids=[agent.agent_id],
    )

    event = LifecycleEvent(
        event_type="fork",
        task_index=task_index,
        agent_ids=[agent.agent_id],
        new_agent_ids=[child_a.agent_id, child_b.agent_id],
        reason=f"Divergent task types ({type_a} vs {type_b}), entropy={entropy:.2f}",
    )

    return [child_a, child_b], event


def try_merge_pool(
    pool: list[Agent],
    task_index: int,
    backbone_llm: str,
) -> tuple[list[Agent], list[LifecycleEvent]]:
    """Check all agent pairs for merge eligibility. Merge the most similar pair."""
    from .agent import Agent as AgentClass, AgentProfile

    events: list[LifecycleEvent] = []
    merged_ids: set[str] = set()

    # Snapshot the pool to avoid indexing issues when pool is modified mid-loop.
    # We iterate over the snapshot, but accumulate which agents to remove/add.
    snapshot = list(pool)
    to_remove: set[str] = set()
    to_add: list[Agent] = []

    for i in range(len(snapshot)):
        for j in range(i + 1, len(snapshot)):
            a, b = snapshot[i], snapshot[j]
            if a.agent_id in merged_ids or b.agent_id in merged_ids:
                continue
            if a.agent_id in to_remove or b.agent_id in to_remove:
                continue
            if a.task_count < MERGE_MIN_TASKS or b.task_count < MERGE_MIN_TASKS:
                continue

            sim = _profile_cosine_similarity(a, b)
            if sim < MERGE_SIMILARITY_THRESHOLD:
                continue

            # Merge: create a new agent that combines both profiles
            merged_profile = _merge_profiles(a.profile, b.profile, backbone_llm)
            merged = AgentClass(
                profile=merged_profile,
                parent_ids=[a.agent_id, b.agent_id],
            )
            merged.task_count = (a.task_count + b.task_count) // 2

            to_remove.update({a.agent_id, b.agent_id})
            to_add.append(merged)
            merged_ids.update({a.agent_id, b.agent_id})

            events.append(LifecycleEvent(
                event_type="merge",
                task_index=task_index,
                agent_ids=[a.agent_id, b.agent_id],
                new_agent_ids=[merged.agent_id],
                reason=f"High profile similarity ({sim:.2f})",
            ))

    # Apply all merges at once (no mid-loop mutation)
    pool = [ag for ag in pool if ag.agent_id not in to_remove]
    pool.extend(to_add)

    return pool, events


def try_prune(agent: Agent, pool_mean_score: float, task_index: int) -> LifecycleEvent | None:
    """Prune agent if it has been consistently underperforming."""
    if agent.consecutive_underperformance >= PRUNE_UNDERPERFORMANCE_TASKS:
        return LifecycleEvent(
            event_type="prune",
            task_index=task_index,
            agent_ids=[agent.agent_id],
            reason=f"Consecutive underperformance for {agent.consecutive_underperformance} tasks",
        )
    return None


def try_genesis(
    pool: list[Agent],
    recent_task_types: list[str],
    task_index: int,
    backbone_llm: str,
    coverage_threshold: float = GENESIS_COVERAGE_THRESHOLD,
) -> tuple[list[Agent], list[LifecycleEvent]]:
    """
    Create new agents for task types that have no adequate pool coverage.
    Spawns from the best generalist agent.
    """
    from .agent import Agent as AgentClass, AgentProfile

    events: list[LifecycleEvent] = []
    new_agents: list[AgentClass] = []

    # Find uncovered task types
    type_counts: dict[str, int] = {}
    for t in recent_task_types[-20:]:
        type_counts[t] = type_counts.get(t, 0) + 1

    for task_type, count in type_counts.items():
        if count < 3:
            continue  # Not enough signal

        # Check if any agent covers this type
        max_affinity = max(
            _compute_affinity(agent, task_type)
            for agent in pool
        )

        if max_affinity >= coverage_threshold:
            continue

        # Find best generalist (agent with most balanced skill_memory)
        generalist = _find_best_generalist(pool)
        if not generalist:
            continue

        new_profile = _seed_profile_for_type(generalist.profile, task_type, backbone_llm)
        new_agent = AgentClass(profile=new_profile, parent_ids=[generalist.agent_id])

        new_agents.append(new_agent)
        events.append(LifecycleEvent(
            event_type="genesis",
            task_index=task_index,
            agent_ids=[generalist.agent_id],
            new_agent_ids=[new_agent.agent_id],
            reason=f"Coverage gap for task type '{task_type}' (max affinity={max_affinity:.2f})",
        ))

    return new_agents, events


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * math.log(p + 1e-10) for p in probs)


def _profile_cosine_similarity(a: Agent, b: Agent) -> float:
    all_domains = sorted(set(a.profile.skill_memory) | set(b.profile.skill_memory))
    if not all_domains:
        return 0.0
    vec_a = [a.profile.skill_memory.get(d, 0.0) for d in all_domains]
    vec_b = [b.profile.skill_memory.get(d, 0.0) for d in all_domains]

    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _compute_affinity(agent: Agent, task_type: str) -> float:
    return agent.profile.skill_memory.get(task_type, 0.2)


def _find_best_generalist(pool: list[Agent]) -> Agent | None:
    """Agent with highest mean skill across all domains (most broadly capable)."""
    if not pool:
        return None
    return max(
        pool,
        key=lambda a: sum(a.profile.skill_memory.values()) / max(len(a.profile.skill_memory), 1),
    )


def _fork_profile(parent_profile: AgentProfile, dominant_type: str, backbone_llm: str) -> AgentProfile:
    """Create a child profile biased toward a specific task type."""
    from .agent import AgentProfile

    child = copy.deepcopy(parent_profile)
    # Boost the dominant type's skill
    child.skill_memory[dominant_type] = min(1.0, child.skill_memory.get(dominant_type, 0.5) + 0.15)
    # Update persona
    child.persona = f"{parent_profile.persona} (specialized toward {dominant_type})"
    # Filter task history to dominant type
    child.task_history = [t for t in parent_profile.task_history if t.get("type") == dominant_type]
    return child


def _merge_profiles(a: AgentProfile, b: AgentProfile, backbone_llm: str) -> AgentProfile:
    """Merge two profiles: take max skill_memory, average perf_stats, combine task_history."""
    from .agent import AgentProfile

    all_domains = set(a.skill_memory) | set(b.skill_memory)
    merged_skills = {
        d: max(a.skill_memory.get(d, 0.0), b.skill_memory.get(d, 0.0))
        for d in all_domains
    }
    merged_perf: dict[str, list[float]] = {}
    for domain in all_domains:
        merged_perf[domain] = (
            a.perf_stats.get(domain, []) + b.perf_stats.get(domain, [])
        )[-20:]

    return AgentProfile(
        persona=f"Merged specialist combining: {a.persona[:100]} AND {b.persona[:100]}",
        skill_memory=merged_skills,
        task_history=(a.task_history + b.task_history)[-20:],
        collab_log={**a.collab_log, **b.collab_log},
        perf_stats=merged_perf,
    )


def _seed_profile_for_type(parent_profile: AgentProfile, task_type: str, backbone_llm: str) -> AgentProfile:
    """Create a new agent profile seeded for a specific task type."""
    from .agent import AgentProfile

    new_skills = {task_type: 0.4}  # Start with some base skill
    return AgentProfile(
        persona=f"A specialist agent being developed for {task_type} tasks, derived from a generalist.",
        skill_memory=new_skills,
        task_history=[],
        collab_log={},
        perf_stats={},
    )
