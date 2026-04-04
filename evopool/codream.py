"""
Co-Dream: Generative creative session between collaborating agents.

Co-Dream is NOT memory copying. It is an offline joint imagination session where
agents that just worked together enter a "dream state" to:

  1. REFLECT   — each agent shares what surprised, confused, or intrigued them
  2. CONTRAST  — agents compare their own approach to the best performer's approach,
                 extracting specific "delta insights" (inspired by MemCollab's contrastive
                 trajectory distillation: arXiv:2603.23234)
  3. IMAGINE   — agents propose hypothetical extensions grounded in the contrast deltas
  4. DEBATE    — agents challenge each other's imaginations (structured adversarial)
  5. CRYSTALLIZE — from the debate, each agent privately distills one novel insight
                   that updates their profile

The asymmetry is preserved through SELECTIVE LISTENING:
  - Agent A only deeply engages with Agent B's ideas in B's strong domains
  - This prevents homogenization: generalists don't become copies of specialists

Key distinctions:
  - MemCollab (2603.23234): contrastive distillation → agent-AGNOSTIC shared memory
    (homogenizes the pool over time)
  - Co-Dream: contrastive grounding + asymmetric debate → PRIVATE divergent insights
    (preserves specialization diversity while still learning from inter-agent differences)

Inspired by:
  - Neuroscience of sleep: offline consolidation + creative reactivation
  - MemCollab contrastive trajectory distillation (phase 2)
  - Jazz improvisation: musicians riff off each other's ideas asymmetrically
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .llm import llm_call

if TYPE_CHECKING:
    from .agent import Agent


# ------------------------------------------------------------------
# Co-Dream session data structures
# ------------------------------------------------------------------

@dataclass
class Reflection:
    """Phase 1: what each agent found surprising/interesting about the task."""
    agent_id: str
    content: str           # free-form reflection text
    surprise_domains: list[str]   # domains where the agent felt challenged/curious


@dataclass
class ContrastInsight:
    """Phase 2 (MemCollab-inspired): what each agent learns by comparing to the best performer."""
    agent_id: str
    best_agent_id: str       # who performed best on this task
    delta_observation: str   # what the best agent likely did differently
    hypothesis: str          # why that worked better


@dataclass
class Imagination:
    """Phase 3: a hypothetical idea proposed by one agent."""
    agent_id: str
    idea: str              # the novel idea / "what if" proposal
    target_domains: list[str]    # which domains this idea touches
    confidence: float      # 0-1: how confident the agent is
    grounded_in_contrast: bool = False  # True if derived from contrast phase


@dataclass
class DebateExchange:
    """Phase 3: one round of challenge/response between agents."""
    challenger_id: str
    target_id: str
    challenge: str         # the challenge or critique
    response: str          # the target's response / refinement


@dataclass
class CrystallizedInsight:
    """Phase 4: what each agent privately takes away from the dream session."""
    agent_id: str
    insight: str           # the novel distilled lesson
    affected_domains: list[str]
    skill_updates: dict[str, float]   # domain -> delta (can be positive or negative)
    new_hypotheses: list[str]         # new beliefs/hypotheses to test in future tasks


@dataclass
class CoDreamSession:
    """Full record of one Co-Dream session."""
    task_id: str
    team_ids: list[str]
    reflections: list[Reflection] = field(default_factory=list)
    contrasts: list[ContrastInsight] = field(default_factory=list)
    imaginations: list[Imagination] = field(default_factory=list)
    debates: list[DebateExchange] = field(default_factory=list)
    insights: list[CrystallizedInsight] = field(default_factory=list)


# ------------------------------------------------------------------
# Main Co-Dream entry point
# ------------------------------------------------------------------

_CODREAM_SUCCESS_THRESHOLD = 0.6  # Only run full Co-Dream on failures; brief on successes

def run_codream(
    team: list[Agent],
    task: dict,
    task_results: dict[str, dict],
    backbone_llm: str,
    mode: str = "asymmetric",    # "asymmetric" | "symmetric" | "none" (ablation)
    strength_threshold: float = 0.55,
    max_imaginations_per_agent: int = 2,
    debate_rounds: int = 1,
) -> CoDreamSession | None:
    """
    Score-gated Co-Dream session:

    - On FAILURE (team_score < threshold): run REFLECT + CRYSTALLIZE only.
      Skips CONTRAST/IMAGINE/DEBATE — those 3 phases add ~10 LLM calls of noise
      from the 8B model. The key signal (what went wrong, how to update skills)
      is captured in REFLECT + CRYSTALLIZE alone.

    - On SUCCESS (team_score >= threshold): run abbreviated REFLECT only
      (log what went right for future reference, no skill updates needed).

    This reduces LLM calls by ~70% vs the 5-phase pipeline and eliminates
    the noise from IMAGINE/DEBATE on a weak 8B model.

    Asymmetric mode (anti-homogenization) is preserved in CRYSTALLIZE:
      Agent A incorporates insights from B only in B's strong domains.
    """
    if mode == "none" or len(team) < 2:
        return None

    task_id = str(task.get("id", "unknown"))
    task_domain = task.get("domain", task.get("type", ""))
    session = CoDreamSession(task_id=task_id, team_ids=[a.agent_id for a in team])

    # Compute team score to decide which path to take
    scores = {aid: r.get("score", 0.0) for aid, r in task_results.items()}
    team_score = sum(scores.values()) / max(len(scores), 1) if scores else 0.0

    # Build shared context
    task_context = _build_task_context(task, task_results)

    # --- Phase 1: REFLECT (always) ---
    reflections = _phase_reflect(team, task_context, task_results, backbone_llm)
    session.reflections = reflections

    if team_score >= _CODREAM_SUCCESS_THRESHOLD:
        # Success path: just log reflections, no skill updates
        # (agents are doing well — don't disturb their profiles)
        return session

    # Failure path: CRYSTALLIZE directly from reflections
    # (skip CONTRAST/IMAGINE/DEBATE — too noisy on 8B for failure analysis)
    insights = _phase_crystallize_from_reflections(
        team=team,
        reflections=reflections,
        task_context=task_context,
        backbone_llm=backbone_llm,
        mode=mode,
        strength_threshold=strength_threshold,
    )
    session.insights = insights

    # Apply insights (domain-constrained)
    _apply_insights(team, insights, task_domain=task_domain)

    return session


def _phase_crystallize_from_reflections(
    team: list[Agent],
    reflections: list[Reflection],
    task_context: str,
    backbone_llm: str,
    mode: str,
    strength_threshold: float,
) -> list[CrystallizedInsight]:
    """
    Simplified crystallize: distill insights directly from reflections,
    without the intermediate CONTRAST/IMAGINE/DEBATE phases.

    Each agent sees: their own reflection + reflections from stronger peers
    (same asymmetric rule as before). Asks: what ONE strategy update do you
    take away from this failure and your teammates' observations?
    """
    insights: list[CrystallizedInsight] = []
    reflection_map = {r.agent_id: r for r in reflections}

    for agent in team:
        my_reflection = reflection_map.get(agent.agent_id)
        if not my_reflection:
            continue

        # Gather peer reflections (asymmetric: from stronger-in-domain peers only)
        peer_material: list[str] = []
        for other in team:
            if other.agent_id == agent.agent_id:
                continue
            peer_refl = reflection_map.get(other.agent_id)
            if not peer_refl:
                continue
            if mode == "asymmetric":
                # Only learn from peers stronger in some domain
                stronger_in = [
                    d for d in other.profile.skill_memory
                    if other.profile.skill_memory.get(d, 0.0) >= strength_threshold
                    and other.profile.skill_memory.get(d, 0.0) > agent.profile.skill_memory.get(d, 0.0)
                ]
                if not stronger_in:
                    continue
            peer_material.append(f"Teammate {other.agent_id[:6]}: {peer_refl.content}")

        prompt = (
            f"You are {agent.profile.persona}\n"
            f"Your current skills: {agent.profile.skill_memory}\n\n"
            f"Task context (you FAILED this): {task_context}\n\n"
            f"Your reflection: {my_reflection.content}\n\n"
            + (f"Peer observations:\n" + "\n".join(peer_material) + "\n\n" if peer_material else "")
            + "Based on this failure and your team's observations, what is ONE concrete strategy "
            "update you are taking away?\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "insight": "one concrete lesson (2 sentences max)",\n'
            '  "affected_domains": ["domain1"],\n'
            '  "skill_updates": {"domain1": 0.05},\n'
            '  "new_hypotheses": ["hypothesis"]\n'
            "}"
        )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=250)
            data = json.loads(raw.strip())
            insights.append(CrystallizedInsight(
                agent_id=agent.agent_id,
                insight=data.get("insight", ""),
                affected_domains=data.get("affected_domains", []),
                skill_updates=data.get("skill_updates", {}),
                new_hypotheses=data.get("new_hypotheses", []),
            ))
        except Exception:
            pass

    return insights


# ------------------------------------------------------------------
# Phase 1: REFLECT
# ------------------------------------------------------------------

def _phase_reflect(
    team: list[Agent],
    task_context: str,
    task_results: dict[str, dict],
    backbone_llm: str,
) -> list[Reflection]:
    """
    Each agent articulates what was surprising, confusing, or intriguing.
    This surfaces latent knowledge gaps and curiosities — the raw material
    for imagination.
    """
    reflections: list[Reflection] = []
    for agent in team:
        score = task_results.get(agent.agent_id, {}).get("score", 0.5)
        prompt = (
            f"You are {agent.profile.persona}\n\n"
            f"You just completed a task as part of a team.\n"
            f"Task context: {task_context}\n"
            f"Your performance score: {score:.2f}\n\n"
            "Reflect honestly on this experience:\n"
            "1. What surprised you or worked differently than expected?\n"
            "2. What knowledge gaps did you notice in yourself?\n"
            "3. What did you find genuinely interesting or curious?\n"
            "4. What domains felt like they were tested? List them.\n\n"
            "Be honest, not diplomatic. Express genuine intellectual curiosity.\n"
            "Respond with JSON:\n"
            "{\n"
            '  "reflection": "your honest reflection (2-4 sentences)",\n'
            '  "surprise_domains": ["domain1", "domain2"]\n'
            "}"
        )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=250)
            data = json.loads(raw.strip())
            reflections.append(Reflection(
                agent_id=agent.agent_id,
                content=data.get("reflection", ""),
                surprise_domains=data.get("surprise_domains", []),
            ))
        except Exception:
            reflections.append(Reflection(
                agent_id=agent.agent_id,
                content="",
                surprise_domains=[task_context.split(".")[0]],
            ))
    return reflections


# ------------------------------------------------------------------
# Phase 2: CONTRAST (MemCollab-inspired contrastive distillation)
# ------------------------------------------------------------------

def _phase_contrast(
    team: list[Agent],
    task_results: dict[str, dict],
    task_context: str,
    reflections: list[Reflection],
    backbone_llm: str,
) -> list[ContrastInsight]:
    """
    MemCollab-inspired phase: each agent compares their own trajectory to the best
    performer's, extracting specific "delta insights" about what worked differently.

    Key difference from MemCollab: these insights are PRIVATE (not shared into a
    common pool), so they don't homogenize the pool — each agent develops a
    distinct interpretation of the same performance gap.
    """
    if not task_results:
        return []

    # Find best-performing agent on this task
    scores = {aid: r.get("score", 0.0) for aid, r in task_results.items()}
    best_aid = max(scores, key=scores.get) if scores else None
    if best_aid is None:
        return []

    best_reflection = next((r for r in reflections if r.agent_id == best_aid), None)
    contrasts: list[ContrastInsight] = []

    for agent in team:
        my_score = scores.get(agent.agent_id, 0.0)
        best_score = scores.get(best_aid, 0.0)

        # Skip if this IS the best agent or scores are very close
        if agent.agent_id == best_aid or abs(best_score - my_score) < 0.05:
            continue

        best_reflection_text = best_reflection.content if best_reflection else "(no reflection)"

        prompt = (
            f"You are {agent.profile.persona}\n\n"
            f"Task context: {task_context}\n"
            f"Your performance score: {my_score:.2f}\n"
            f"Best teammate's score: {best_score:.2f} (Agent {best_aid[:6]})\n\n"
            f"The best performer reflected: \"{best_reflection_text}\"\n\n"
            "Analyze the performance gap. What might the top performer have done DIFFERENTLY?\n"
            "Be concrete and specific — not 'they were better', but 'they likely did X\n"
            "because of their reflection mentioning Y'.\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "delta": "What they likely did differently (1-2 sentences)",\n'
            '  "hypothesis": "Why that approach worked better (1 sentence)"\n'
            "}"
        )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=200)
            data = json.loads(raw.strip())
            contrasts.append(ContrastInsight(
                agent_id=agent.agent_id,
                best_agent_id=best_aid,
                delta_observation=data.get("delta", ""),
                hypothesis=data.get("hypothesis", ""),
            ))
        except Exception:
            pass

    return contrasts


# ------------------------------------------------------------------
# Phase 3: IMAGINE (grounded by contrast deltas)
# ------------------------------------------------------------------

def _phase_imagine(
    team: list[Agent],
    reflections: list[Reflection],
    contrasts: list[ContrastInsight],
    task_context: str,
    backbone_llm: str,
) -> list[Imagination]:
    """
    Agents read each other's reflections and propose NEW ideas / "what if" extensions.
    Contrast deltas from phase 2 provide grounding so ideas are specific, not vague.
    """
    reflections_text = "\n".join(
        f"Agent {r.agent_id} reflects: {r.content}" for r in reflections
    )
    contrasts_by_agent = {c.agent_id: c for c in contrasts}
    imaginations: list[Imagination] = []

    for agent in team:
        my_contrast = contrasts_by_agent.get(agent.agent_id)
        contrast_section = ""
        if my_contrast:
            contrast_section = (
                f"\nFrom comparing to the top performer:\n"
                f"- What they did differently: {my_contrast.delta_observation}\n"
                f"- Your hypothesis: {my_contrast.hypothesis}\n"
            )

        prompt = (
            f"You are {agent.profile.persona}\n\n"
            f"Task context: {task_context}\n\n"
            f"Your team's reflections after the task:\n{reflections_text}\n"
            f"{contrast_section}\n"
            "You are now in a DREAM STATE — free from the constraints of the task.\n"
            "Propose 1-2 bold ideas or hypotheses sparked by these reflections and your\n"
            "contrast analysis. Ground at least one idea in what you observed about the\n"
            "performance gap (if any).\n\n"
            "Good ideas are:\n"
            "- 'What if we decomposed this differently...'\n"
            "- 'I wonder if combining X and Y approaches would...'\n"
            "- 'The contrast suggests trying Z, but extended to...'\n"
            "- 'A completely different strategy could be...'\n\n"
            "Be imaginative. Speculate. These are hypotheses, not commitments.\n"
            "Respond with JSON:\n"
            "{\n"
            '  "ideas": [\n'
            '    {"idea": "...", "domains": ["domain1"], "confidence": 0.6, "from_contrast": true},\n'
            '    {"idea": "...", "domains": ["domain2"], "confidence": 0.4, "from_contrast": false}\n'
            "  ]\n"
            "}"
        )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=400)
            data = json.loads(raw.strip())
            for item in data.get("ideas", [])[:2]:
                imaginations.append(Imagination(
                    agent_id=agent.agent_id,
                    idea=item.get("idea", ""),
                    target_domains=item.get("domains", []),
                    confidence=float(item.get("confidence", 0.5)),
                    grounded_in_contrast=bool(item.get("from_contrast", False)),
                ))
        except Exception:
            pass

    return imaginations


# ------------------------------------------------------------------
# Phase 3: DEBATE
# ------------------------------------------------------------------

def _phase_debate(
    team: list[Agent],
    imaginations: list[Imagination],
    task_context: str,
    backbone_llm: str,
    mode: str,
    strength_threshold: float,
) -> list[DebateExchange]:
    """
    Agents challenge each other's imaginations.

    Asymmetric rule: Agent A engages deeply with Agent B's idea ONLY if
    the idea's target domains align with B's strong domains. This prevents
    weak ideas from spreading — agents don't waste energy debating outside
    their competence.

    The debate sharpens ideas: challenges force the proposer to refine or
    abandon their imagination, producing a stronger or clearer crystallization.
    """
    debates: list[DebateExchange] = []
    imagination_text_all = "\n".join(
        f"Agent {im.agent_id}: [{', '.join(im.target_domains)}] {im.idea}"
        for im in imaginations
    )

    for challenger in team:
        for imagination in imaginations:
            if imagination.agent_id == challenger.agent_id:
                continue  # Don't debate your own ideas

            # Asymmetric: challenger only debates if imagination touches domains
            # where the proposer is demonstrably stronger than challenger
            if mode == "asymmetric":
                proposer = _find_agent(team, imagination.agent_id)
                if proposer is None:
                    continue
                relevant_domains = [
                    d for d in imagination.target_domains
                    if proposer.profile.skill_memory.get(d, 0.0) >= strength_threshold
                    and proposer.profile.skill_memory.get(d, 0.0)
                    > challenger.profile.skill_memory.get(d, 0.0)
                ]
                if not relevant_domains:
                    continue  # Skip: challenger is not learning anything useful here

            # Build challenge
            challenge_prompt = (
                f"You are {challenger.profile.persona}\n\n"
                f"Task context: {task_context}\n\n"
                f"Your colleague Agent {imagination.agent_id} proposed this idea:\n"
                f'"{imagination.idea}"\n\n'
                "As a critical peer, challenge this idea constructively:\n"
                "- What assumption is it making that might be wrong?\n"
                "- What edge case or failure mode does it ignore?\n"
                "- OR: How would you refine/extend it to make it stronger?\n\n"
                "Be intellectually honest. One focused challenge (2-3 sentences)."
            )
            try:
                challenge = llm_call(
                    model=backbone_llm, user=challenge_prompt, max_tokens=200
                )
            except Exception:
                continue

            # Build response from the proposer
            proposer = _find_agent(team, imagination.agent_id)
            if proposer is None:
                continue
            response_prompt = (
                f"You are {proposer.profile.persona}\n\n"
                f"You proposed: \"{imagination.idea}\"\n\n"
                f"Your colleague Agent {challenger.agent_id} challenges:\n{challenge}\n\n"
                "Respond: either refine your idea in light of this challenge, "
                "or explain why the challenge misses the point. "
                "Be honest — if the challenge is valid, update your idea. "
                "2-3 sentences."
            )
            try:
                response = llm_call(
                    model=backbone_llm, user=response_prompt, max_tokens=200
                )
                debates.append(DebateExchange(
                    challenger_id=challenger.agent_id,
                    target_id=imagination.agent_id,
                    challenge=challenge,
                    response=response,
                ))
            except Exception:
                continue

    return debates


# ------------------------------------------------------------------
# Phase 4: CRYSTALLIZE
# ------------------------------------------------------------------

def _phase_crystallize(
    team: list[Agent],
    imaginations: list[Imagination],
    debates: list[DebateExchange],
    task_context: str,
    backbone_llm: str,
    mode: str,
    strength_threshold: float,
) -> list[CrystallizedInsight]:
    """
    Each agent privately distills one novel insight from the dream session.

    This is the most important phase. The insight should be something the agent
    did NOT know before the session — a new hypothesis, a new approach, a new
    belief. It updates the agent's private profile.

    Asymmetric filtering: Agent A only incorporates insights from B's ideas in
    B's strong domains (same rule as debate). This is what prevents homogenization.
    """
    insights: list[CrystallizedInsight] = []
    imaginations_by_agent = {im.agent_id: im for im in imaginations}

    for agent in team:
        # Gather the ideas and debates that are RELEVANT to this agent
        # (asymmetric: only deep-engage with ideas from stronger-in-domain peers)
        relevant_material: list[str] = []

        for im in imaginations:
            if im.agent_id == agent.agent_id:
                continue  # Skip own ideas (agent already knows them)

            if mode == "asymmetric":
                proposer = _find_agent(team, im.agent_id)
                if proposer is None:
                    continue
                relevant = [
                    d for d in im.target_domains
                    if proposer.profile.skill_memory.get(d, 0.0) >= strength_threshold
                    and proposer.profile.skill_memory.get(d, 0.0)
                    > agent.profile.skill_memory.get(d, 0.0)
                ]
                if not relevant:
                    continue  # Skip: this agent is not learning from this proposer's domain
                relevant_material.append(
                    f"[From stronger peer {im.agent_id} on {relevant}]: {im.idea}"
                )
            else:
                # Symmetric: engage with everything
                relevant_material.append(f"[From {im.agent_id}]: {im.idea}")

        # Add relevant debate exchanges
        for debate in debates:
            if debate.target_id == agent.agent_id:
                relevant_material.append(
                    f"[Critique of your idea from {debate.challenger_id}]: {debate.challenge}"
                )
            elif debate.challenger_id == agent.agent_id:
                relevant_material.append(
                    f"[Your challenge was responded to by {debate.target_id}]: {debate.response}"
                )

        if not relevant_material:
            continue  # Nothing to crystallize

        material_text = "\n".join(relevant_material)

        prompt = (
            f"You are {agent.profile.persona}\n"
            f"Your current skill profile: {agent.profile.skill_memory}\n\n"
            f"Task context: {task_context}\n\n"
            f"Ideas and exchanges from your team's dream session:\n{material_text}\n\n"
            "You are crystallizing your private takeaway from this dream session.\n"
            "What is ONE novel insight you are taking away?\n"
            "This should be something you DID NOT know before — a new hypothesis,\n"
            "a new technique, a new belief, or a new approach.\n\n"
            "Also: which of your skill areas does this update? By how much? (−0.1 to +0.2)\n"
            "And: what new hypothesis will you test in future tasks?\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "insight": "...",\n'
            '  "affected_domains": ["domain1"],\n'
            '  "skill_updates": {"domain1": 0.05, "domain2": -0.02},\n'
            '  "new_hypotheses": ["hypothesis1", "hypothesis2"]\n'
            "}"
        )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=400)
            data = json.loads(raw.strip())
            insights.append(CrystallizedInsight(
                agent_id=agent.agent_id,
                insight=data.get("insight", ""),
                affected_domains=data.get("affected_domains", []),
                skill_updates=data.get("skill_updates", {}),
                new_hypotheses=data.get("new_hypotheses", []),
            ))
        except Exception:
            pass

    return insights


# ------------------------------------------------------------------
# Apply insights to agent profiles
# ------------------------------------------------------------------

def _apply_insights(
    team: list[Agent],
    insights: list[CrystallizedInsight],
    task_domain: str = "",
) -> None:
    """
    Apply crystallized insights to agent profiles.
    Skill updates are bounded: max +0.15, min −0.1 per domain per session.
    New hypotheses are stored as strings in a new 'hypotheses' field.

    Domain-constraint: if task_domain is given, only update skills relevant
    to that domain. This prevents cross-domain skill corruption where e.g.
    a math task causes agents to lower their code skill (catastrophic forgetting).
    """
    agent_map = {a.agent_id: a for a in team}
    for insight in insights:
        agent = agent_map.get(insight.agent_id)
        if agent is None:
            continue

        # Apply skill updates (bounded, domain-constrained)
        for domain, delta in insight.skill_updates.items():
            # Skip updates to domains unrelated to the current task's domain.
            # This is the key fix for cross-domain profile corruption.
            if task_domain and not _domains_related(domain, task_domain):
                continue
            delta = max(-0.10, min(0.15, delta))  # bound per-session update
            old = agent.profile.skill_memory.get(domain, 0.3)
            agent.profile.skill_memory[domain] = max(0.0, min(1.0, old + delta))

        # Store new hypotheses (ephemeral — used in next task's system prompt)
        if not hasattr(agent.profile, 'hypotheses'):
            agent.profile.hypotheses = []
        agent.profile.hypotheses = (
            getattr(agent.profile, 'hypotheses', []) + insight.new_hypotheses
        )[-5:]  # Keep last 5 hypotheses max

        # Update collab_log with the insight (what was learned from this dream)
        for aid in [i.agent_id for i in insights if i.agent_id != insight.agent_id]:
            if aid not in agent.profile.collab_log:
                agent.profile.collab_log[aid] = []
            agent.profile.collab_log[aid].append(insight.insight[:150])
            agent.profile.collab_log[aid] = agent.profile.collab_log[aid][-5:]


# Domain relatedness map — skills that belong to the same macro-domain cluster.
# Co-Dream should only update skills within the same cluster as the current task.
_DOMAIN_CLUSTERS = {
    "math": {"math", "gsm8k", "algebra", "calculus", "math_competition", "math_word_problem",
             "arithmetic", "math_competition", "number_theory", "geometry", "precalculus"},
    "code": {"code", "mbpp", "humaneval", "programming", "code_completion", "code_generation",
             "coding", "software", "implementation"},
    "qa":   {"qa", "hotpotqa", "drop", "reading_comprehension", "multi_hop_qa", "factual_qa",
             "reading", "comprehension"},
    "general": {"general", "reasoning", "logic", "analysis"},
}

def _domains_related(update_domain: str, task_domain: str) -> bool:
    """
    Return True if update_domain is semantically related to task_domain.
    Uses both the legacy hardcoded cluster map (fast) and semantic embedding
    similarity (oracle-free, generalizable) as a fallback.
    """
    update_d = update_domain.lower().replace("-", "_").replace(" ", "_")
    task_d = task_domain.lower().replace("-", "_").replace(" ", "_")

    # Fast path: check hardcoded cluster map first
    for cluster in _DOMAIN_CLUSTERS.values():
        if task_d in cluster:
            return update_d in cluster or update_d == task_d

    # Slow path: semantic embedding similarity (for unknown domains)
    try:
        from .task_embed import embed, cosine_sim
        sim = cosine_sim(embed(update_d), embed(task_d))
        return sim >= 0.20
    except Exception:
        return True  # conservative fallback


# ------------------------------------------------------------------
# Diversity metrics (unchanged interface, updated internals)
# ------------------------------------------------------------------

def compute_profile_diversity(agents: list[Agent]) -> float:
    """Mean pairwise cosine distance of agent skill profiles. Higher = more diverse."""
    if len(agents) < 2:
        return 0.0
    all_domains = sorted({d for a in agents for d in a.profile.skill_memory})
    if not all_domains:
        return 0.0
    vectors = [
        [a.profile.skill_memory.get(d, 0.0) for d in all_domains]
        for a in agents
    ]
    distances = [
        _cosine_distance(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return sum(distances) / len(distances)


def compute_skill_entropy(agents: list[Agent]) -> float:
    """Entropy of the pool's aggregate skill distribution."""
    if not agents:
        return 0.0
    all_domains = sorted({d for a in agents for d in a.profile.skill_memory})
    if not all_domains:
        return 0.0
    mean_skills = [
        sum(a.profile.skill_memory.get(d, 0.0) for a in agents) / len(agents)
        for d in all_domains
    ]
    total = sum(mean_skills)
    if total == 0:
        return 0.0
    probs = [s / total for s in mean_skills]
    return -sum(p * math.log(p + 1e-10) for p in probs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_task_context(task: dict, task_results: dict[str, dict]) -> str:
    scores = {aid: r.get("score", 0.5) for aid, r in task_results.items()}
    score_str = ", ".join(f"{aid}: {s:.2f}" for aid, s in scores.items())
    return f"Type: {task.get('type', 'general')}. Scores: {score_str}."


def _find_agent(team: list[Agent], agent_id: str) -> Agent | None:
    for a in team:
        if a.agent_id == agent_id:
            return a
    return None


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)
