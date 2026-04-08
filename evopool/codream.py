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
import re as _re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .llm import llm_call


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown code fences if present."""
    raw = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` wrappers
    raw = _re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = _re.sub(r"\n?```\s*$", "", raw)
    # Extract the outermost { ... } block if there's leading/trailing text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    return json.loads(raw)

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
    is_generalizable: bool = False    # True = cross-domain reasoning strategy (apply everywhere)
    # Fix 1 (scope tagging): explicit transferability classification prevents premature
    # generalization where sub-domain insights (e.g. "algebra: simplify fractions first")
    # pollute agents solving geometry or combinatorics tasks via persona injection.
    transferability: str = "general"  # "general" | "subdomain" | "task_specific"
    domain_scope: str = "any"         # e.g. "algebra", "geometry", "code_generation", "any"


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
    # Verify stats (populated when evaluator_fn is used)
    n_insights_generated: int = 0    # insights before verify
    n_insights_verified: int = 0     # insights that passed verify
    verify_rate: float = 0.0         # n_verified / n_generated (0 if n_generated==0)


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
    evaluator_fn=None,           # optional: fn(task, response) -> float, for insight verification
    disable_l3: bool = False,    # E22 ablation: no cross-domain L3 broadcast
    disable_l2: bool = False,    # E23 ablation: no subdomain L2 accumulation
    enhanced: bool = False,      # E25: disagreement trigger + success extraction + domain_general
) -> CoDreamSession | None:
    """
    Score-gated Co-Dream session with insight verification.

    - On FAILURE (team_score < threshold): run REFLECT + CRYSTALLIZE.
      If evaluator_fn is provided, each insight is verified by re-attempting
      the failing task with the updated persona. Only insights that improve
      the score are applied (gating noisy/useless insights out).
      The verification score is NOT counted in the benchmark — it is used
      solely as a gate to decide whether to keep the insight.

    - On SUCCESS (team_score >= threshold): log reflections only, no updates.

    Additionally, crystallize uses the agent's recent domain failure history
    (from task_history) so insights are grounded in patterns, not single points.

    Asymmetric mode (anti-homogenization) is preserved in CRYSTALLIZE:
      Agent A incorporates insights from B only in B's strong domains.
    """
    if mode == "none" or len(team) < 2:
        return None

    task_id = str(task.get("id", "unknown"))
    task_domain = task.get("domain", task.get("type", ""))
    session = CoDreamSession(task_id=task_id, team_ids=[a.agent_id for a in team])

    # Compute per-agent scores and team average
    scores = {aid: r.get("score", 0.0) for aid, r in task_results.items()}
    team_score = sum(scores.values()) / max(len(scores), 1) if scores else 0.0

    # Disagreement detection (E25 enhanced mode):
    # When some agents succeed and others fail on the SAME task (max-min >= 0.5),
    # this is the richest learning signal for independent tasks — the failing agent
    # can learn from the successful agent's strategy via CONTRAST.
    # E.g. [0,1,1] team: avg=0.67 > threshold, but the 0-agent is missing something.
    score_vals = list(scores.values())
    disagreement = (
        enhanced
        and len(score_vals) > 1
        and (max(score_vals) - min(score_vals)) >= 0.5
    )

    # Fast-path: skip only if team succeeded AND no disagreement between agents
    if team_score >= _CODREAM_SUCCESS_THRESHOLD and not disagreement:
        return session

    # Build shared context (include recent failure history for pattern-grounded crystallize)
    task_context = _build_task_context(task, task_results)

    # --- Full 5-phase CoDream pipeline ---

    # Phase 1: REFLECT — each agent diagnoses what went wrong/right
    reflections = _phase_reflect(team, task_context, task_results, backbone_llm)
    session.reflections = reflections

    # Phase 2: CONTRAST — failing agents compare approach to best performer
    contrasts = _phase_contrast(team, task_results, task_context, reflections, backbone_llm)
    session.contrasts = contrasts

    # Phase 3: IMAGINE — agents propose strategy transfers grounded in contrast
    imaginations = _phase_imagine(team, reflections, contrasts, task_context, backbone_llm)
    session.imaginations = imaginations

    # Phase 4: DEBATE — peers challenge each other's proposals (1 round)
    debates = _phase_debate(team, imaginations, task_context, backbone_llm, mode, strength_threshold)
    session.debates = debates

    # Phase 5: CRYSTALLIZE — distill surviving proposals into structured insights
    insights = _phase_crystallize(
        team=team,
        imaginations=imaginations,
        debates=debates,
        task_context=task_context,
        backbone_llm=backbone_llm,
        mode=mode,
        strength_threshold=strength_threshold,
    )
    session.insights = insights

    # Verify each insight before applying: re-attempt the failing task with the
    # updated persona. Only keep insights that actually improve the score.
    # The re-attempt score is NOT reported to the benchmark — purely a gate.
    if evaluator_fn is not None:
        verified_insights = _verify_insights(
            team=team,
            insights=insights,
            task=task,
            backbone_llm=backbone_llm,
            evaluator_fn=evaluator_fn,
            original_scores=scores,
        )
        session.n_insights_generated = len(insights)
        session.n_insights_verified = len(verified_insights)
        session.verify_rate = (
            len(verified_insights) / len(insights) if insights else 0.0
        )
        session.insights = verified_insights
        _apply_insights(team, verified_insights, task_domain=task_domain,
                        disable_l3=disable_l3, disable_l2=disable_l2)
    else:
        session.n_insights_generated = len(insights)
        session.n_insights_verified = len(insights)  # no verify = all applied
        session.verify_rate = 1.0
        _apply_insights(team, insights, task_domain=task_domain,
                        disable_l3=disable_l3, disable_l2=disable_l2)

    return session


def _verify_insights(
    team: list[Agent],
    insights: list[CrystallizedInsight],
    task: dict,
    backbone_llm: str,
    evaluator_fn,
    original_scores: dict[str, float],
) -> list[CrystallizedInsight]:
    """
    Verify each insight by re-attempting the failing task with updated persona.
    Keep insight only if re-attempt score > original score.

    No data leakage: the re-attempt score is used only as a binary gate
    (keep/discard the insight), not counted in the benchmark evaluation.
    If re-attempt fails to improve, the insight is silently discarded and
    the persona is reverted — as if the insight never happened.
    """
    agent_map = {a.agent_id: a for a in team}
    verified: list[CrystallizedInsight] = []

    for insight in insights:
        if not insight.insight:
            continue
        agent = agent_map.get(insight.agent_id)
        if agent is None:
            continue
        original_score = original_scores.get(insight.agent_id, 0.0)
        # Skip verification for agents that already succeeded
        if original_score >= 1.0:
            verified.append(insight)
            continue

        # Temporarily apply the insight to persona.
        # Also save/restore working_memory: execute_task clears it as a one-shot
        # mechanism, but working memory set by a PREVIOUS task must survive through
        # verification so it is available for the next REAL task execution.
        old_persona = agent.profile.persona
        old_wm = list(getattr(agent.profile, 'working_memory', []))
        agent.profile.persona = (
            old_persona + f"\n[Strategy update from recent failure: {insight.insight}]"
        )
        try:
            result = agent.execute_task(task, backbone_llm)
            new_score = evaluator_fn(task, result.get("response", ""))
            if new_score > original_score:
                # Insight helps on the failing case → keep it
                verified.append(insight)
            else:
                # Insight doesn't help → revert persona, discard insight
                agent.profile.persona = old_persona
        except Exception:
            # Any error → revert and discard
            agent.profile.persona = old_persona
        finally:
            # Always restore working memory — verification must not consume it
            agent.profile.working_memory = old_wm

    return verified


def _phase_crystallize_from_reflections(
    team: list[Agent],
    reflections: list[Reflection],
    task_context: str,
    backbone_llm: str,
    mode: str,
    strength_threshold: float,
    scores: dict | None = None,
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

        # Build recent failure history in this domain for pattern grounding
        domain = task_context.split(".")[0].replace("Type: ", "")
        recent_failures = [
            t for t in agent.profile.task_history[-10:]
            if t.get("score", 1.0) < 0.5 and t.get("type", "") == domain
        ]
        failure_history_str = ""
        if len(recent_failures) >= 2:
            failure_history_str = (
                f"Your recent failures in this domain ({len(recent_failures)} out of last "
                f"{min(10, len(agent.profile.task_history))} tasks): "
                + "; ".join(t.get("type", "?") for t in recent_failures[-3:])
                + "\n\n"
            )

        my_score = scores.get(agent.agent_id, 0.0) if scores else 0.0
        agent_succeeded = my_score >= 0.7

        if agent_succeeded:
            # Success extraction: articulate what worked for reuse by teammates
            prompt = (
                f"You are {agent.profile.persona}\n"
                f"Your current skills: {agent.profile.skill_memory}\n\n"
                f"Task context (you SUCCEEDED, score: {my_score:.1f}): {task_context}\n\n"
                f"Your reflection: {my_reflection.content}\n\n"
                + (f"Teammate(s) FAILED. Their observations:\n"
                   + "\n".join(peer_material) + "\n\n" if peer_material else "")
                + "You solved this while some teammates did not. Articulate the KEY STRATEGY "
                "that made the difference — a SPECIFIC technique, formula, or procedure.\n"
                "Do NOT give generic advice like 'validate carefully' or 'be systematic'.\n"
                "Instead: 'When [pattern], use [technique]: [concrete steps]'\n\n"
                "Classify using FOUR levels:\n"
                '  - "general": works on ANY task (coding, QA, math, anything)\n'
                '  - "domain_general": useful for ALL tasks in this broad domain\n'
                '    (e.g. "for any math: verify by substitution" → domain_general=math)\n'
                '    (e.g. "for any code: check edge cases first" → domain_general=code)\n'
                '  - "subdomain": only a specific sub-topic in this domain\n'
                '  - "task_specific": only this exact problem\n\n'
                "Q1: Help ANY coding task? Q2: Help ANY multi-hop QA? Q3: Help ANY math?\n"
                'All YES → "general" | Same domain only → "domain_general" | '
                'Sub-topic only → "subdomain" | Just this → "task_specific"\n\n'
                "Respond with JSON:\n"
                '{"insight": "strategy (2 sentences max)", '
                '"transferability": "general"|"domain_general"|"subdomain"|"task_specific", '
                '"domain_scope": "any" or "<domain>" or "<subtopic>", '
                '"is_generalizable": true/false, "affected_domains": [...], '
                '"skill_updates": {"domain": 0.05}, "new_hypotheses": ["..."]}'
            )
        else:
            # Failure crystallize: what went wrong and how to fix it
            prompt = (
                f"You are {agent.profile.persona}\n"
                f"Your current skills: {agent.profile.skill_memory}\n\n"
                f"Task context (you FAILED, score: {my_score:.1f}): {task_context}\n\n"
                + failure_history_str
                + f"Your reflection: {my_reflection.content}\n\n"
                + (f"Peer observations (some may have succeeded):\n"
                   + "\n".join(peer_material) + "\n\n" if peer_material else "")
                + "What is ONE concrete strategy you will change after this failure?\n"
                "Be SPECIFIC: name the exact technique that failed and the replacement.\n"
                "Do NOT give generic advice like 'be more careful' or 'validate thoroughly'.\n"
                "Instead: 'When [pattern], do NOT [wrong approach], instead use [specific fix]'\n\n"
                "Classify using FOUR levels:\n"
                '  - "general": metacognitive strategy that works on ANY task type\n'
                '  - "domain_general": useful for ALL tasks in this broad domain\n'
                '    (e.g. "for all math: try special cases first" → domain_general=math)\n'
                '  - "subdomain": only a SPECIFIC sub-topic in this domain\n'
                '  - "task_specific": only this exact problem type\n\n'
                "Q1: Help ANY coding task? Q2: Help ANY QA task? Q3: Help ANY math task?\n"
                'All YES → "general" | Same domain only → "domain_general" | '
                'Sub-topic only → "subdomain" | Just this → "task_specific"\n\n'
                "Example: 'For fraction problems, simplify first' → subdomain=fractions\n"
                "Example: 'For ANY math: verify answer by substitution' → domain_general, scope=math\n"
                "Example: 'Read question twice before answering' → general\n\n"
                "Respond with JSON:\n"
                '{"insight": "lesson (2 sentences max)", '
                '"transferability": "general"|"domain_general"|"subdomain"|"task_specific", '
                '"domain_scope": "any" or "<domain>" or "<subtopic>", '
                '"is_generalizable": true/false, "affected_domains": [...], '
                '"skill_updates": {"domain": 0.05}, "new_hypotheses": ["..."]}'
            )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=350)
            data = _parse_json(raw)
            transferability = data.get("transferability", "general")
            domain_scope = data.get("domain_scope", "any")
            # Enforce consistency: subdomain/task_specific insights are NOT generalizable
            is_gen = (transferability == "general")
            insights.append(CrystallizedInsight(
                agent_id=agent.agent_id,
                insight=data.get("insight", ""),
                is_generalizable=is_gen,
                transferability=transferability,
                domain_scope=domain_scope,
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
            f"You just FAILED a task (score: {score:.2f}) as part of a team.\n"
            f"Task context: {task_context}\n\n"
            "Diagnose your failure honestly and concisely:\n"
            "1. What specific step went wrong? (e.g., misread the question, wrong formula, "
            "formatting error, reasoning gap)\n"
            "2. What would you do differently next time to avoid this mistake?\n"
            "3. Is this a general reasoning error (applies to many task types) or "
            "specific to this sub-domain?\n"
            "4. What domain(s) does this failure relate to?\n\n"
            "Be concrete — avoid vague answers like 'I need to think more carefully'.\n"
            "Respond with JSON:\n"
            "{\n"
            '  "reflection": "diagnosis + concrete fix (2-3 sentences)",\n'
            '  "surprise_domains": ["domain1"]\n'
            "}"
        )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=250)
            data = _parse_json(raw)
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
            data = _parse_json(raw)
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
            "Based on the reflections and contrast analysis, propose 1-2 CONCRETE "
            "technique improvements you want to try on future similar problems.\n\n"
            "Requirements:\n"
            "- Each idea must name a SPECIFIC mathematical technique, algorithm, or approach\n"
            "- Must explain WHEN to use it (what problem pattern)\n"
            "- Must explain HOW to apply it (concrete steps)\n"
            "- Do NOT propose vague ideas like 'be more systematic' or 'validate better'\n\n"
            "Good examples:\n"
            "- 'For problems with nested absolute values, expand cases by sign regions'\n"
            "- 'When a combinatorics problem involves circular arrangements, fix one element'\n"
            "- 'The contrast shows the winner used generating functions instead of direct counting'\n\n"
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
            data = _parse_json(raw)
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
            "Crystallize ONE concrete, actionable insight from this session.\n\n"
            "REQUIREMENTS for the insight:\n"
            "- Must be a SPECIFIC technique, formula, or step-by-step procedure\n"
            "- Must include WHEN to apply it (what problem pattern triggers it)\n"
            "- Must NOT be generic advice like 'validate carefully' or 'ensure robustness'\n"
            "- Good example: 'When a problem involves modular arithmetic with prime p, "
            "try Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) to simplify exponents'\n"
            "- Bad example: 'Ensure holistic validation across the computation chain'\n\n"
            "Also: which skill area does this update? By how much? (−0.1 to +0.2)\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "insight": "When [specific pattern], use [specific technique]: [concrete steps]",\n'
            '  "affected_domains": ["domain1"],\n'
            '  "skill_updates": {"domain1": 0.05, "domain2": -0.02},\n'
            '  "new_hypotheses": ["hypothesis1"]\n'
            "}"
        )
        try:
            raw = llm_call(model=backbone_llm, user=prompt, max_tokens=400)
            data = _parse_json(raw)
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
    disable_l3: bool = False,
    disable_l2: bool = False,
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

        # Generalizable insights: cross-domain metacognitive strategies that apply to all tasks.
        # ONLY write to persona if transferability=="general" (Fix 1: stricter scope check).
        # Subdomain insights (e.g. "for algebra: simplify fractions first") must NOT pollute
        # the persona — they would be applied to geometry/code/QA tasks where they are wrong.
        if insight.is_generalizable and insight.transferability == "general" and insight.insight:
            if disable_l3:
                continue  # E22 ablation: skip L3 broadcast entirely
            # Fix 2a: Broadcast general metacognitive insights to ALL team members.
            # A general strategy (e.g. "decompose into sub-problems then verify each step")
            # is equally valuable for every agent — not just the one who discovered it.
            # _append_to_persona handles deduplication (Fix 2b) and the 3-slot cap per agent.
            for recipient in team:
                _append_to_persona(recipient, insight.insight, tag="[General strategy]")
            # Track hypothesis only on originating agent (they discovered it)
            if not hasattr(agent.profile, 'hypotheses'):
                agent.profile.hypotheses = []
            agent.profile.hypotheses = (
                getattr(agent.profile, 'hypotheses', []) + [insight.insight]
            )[-5:]
            continue

        # domain_general insights (L2.5): useful for ALL tasks in the same broad domain.
        # E.g. "for any math: verify answer by substitution" applies to all MATH tasks.
        # Stored per top-level domain (math/code/qa), injected whenever task is in that domain.
        if insight.transferability == "domain_general" and insight.insight and not disable_l2:
            if not hasattr(agent.profile, 'domain_insights'):
                agent.profile.domain_insights = {}
            # scope = top-level domain name (e.g. "math", "code", "qa")
            scope = insight.domain_scope.lower().strip() or task_domain.lower().strip()
            # Normalize to top-level domain using cluster map
            for cluster_name, cluster_set in _DOMAIN_CLUSTERS.items():
                if scope in cluster_set or scope == cluster_name:
                    scope = cluster_name
                    break
            existing = agent.profile.domain_insights.get(scope, [])
            fingerprint = insight.insight.strip().lower()[:40]
            if not any(fingerprint in s.lower() for s in existing):
                agent.profile.domain_insights[scope] = (existing + [insight.insight])[-3:]

        # Subdomain-scoped insights: store in agent.profile.subdomain_insights dict.
        # These will be injected into the task prompt ONLY when the task sub-domain matches.
        if insight.transferability == "subdomain" and insight.insight and insight.domain_scope not in ("any", "general", "") and not disable_l2:
            if not hasattr(agent.profile, 'subdomain_insights'):
                agent.profile.subdomain_insights = {}
            scope = insight.domain_scope.lower().replace(" ", "_").replace("-", "_")
            existing = agent.profile.subdomain_insights.get(scope, [])
            # Deduplication: skip if very similar insight already stored for this scope
            fingerprint = insight.insight.strip().lower()[:40]
            if any(fingerprint in s.lower() for s in existing):
                existing = existing  # no-op; fall through to skill update
            else:
                # Keep at most 3 scoped insights per subdomain to avoid prompt bloat
                agent.profile.subdomain_insights[scope] = (existing + [insight.insight])[-3:]
            # Also update skill_memory within the domain (fall through to skill update below)

        # Fix 3: Working memory for task_specific insights.
        # Insights that are only valid for this exact problem type (e.g. "this specific ODE
        # needs integrating factor") are too narrow for persona/subdomain storage. However,
        # they may still be useful for the VERY NEXT task if it shares the same macro-domain.
        # Store as (domain, insight) tuples; agent.py injects and clears on the next task.
        if insight.transferability == "task_specific" and insight.insight and task_domain:
            if not hasattr(agent.profile, 'working_memory'):
                agent.profile.working_memory = []
            agent.profile.working_memory = (
                agent.profile.working_memory + [(task_domain, insight.insight)]
            )[-2:]  # at most 2 one-shot hints; cleared by execute_subtask after use

        # Domain-specific insights: apply skill updates (bounded, domain-constrained)
        for domain, delta in insight.skill_updates.items():
            # Skip updates to domains unrelated to the current task's domain.
            if task_domain and not _domains_related(domain, task_domain):
                continue
            delta = max(-0.10, min(0.15, delta))  # bound per-session update
            old = agent.profile.skill_memory.get(domain, 0.3)
            agent.profile.skill_memory[domain] = max(0.0, min(1.0, old + delta))

        # Store new hypotheses
        if not hasattr(agent.profile, 'hypotheses'):
            agent.profile.hypotheses = []
        agent.profile.hypotheses = (
            getattr(agent.profile, 'hypotheses', []) + insight.new_hypotheses
        )[-5:]

        # Update collab_log with the insight (what was learned from this dream)
        for aid in [i.agent_id for i in insights if i.agent_id != insight.agent_id]:
            if aid not in agent.profile.collab_log:
                agent.profile.collab_log[aid] = []
            agent.profile.collab_log[aid].append(insight.insight[:150])
            agent.profile.collab_log[aid] = agent.profile.collab_log[aid][-5:]


def _append_to_persona(agent, insight_text: str, tag: str = "") -> None:
    """Append a verified insight to agent persona (bounded to last 3 general strategies).

    Fix 2b: Deduplication — skip if an identical or near-identical insight is already
    present in the persona (first 40 chars lowercased match).  Prevents the same
    general strategy from consuming all 3 persona slots when multiple agents or
    multiple tasks produce equivalent insights.
    """
    # Deduplication: skip if very similar insight already in persona
    fingerprint = insight_text.strip().lower()[:40]
    if fingerprint and fingerprint in agent.profile.persona.lower():
        return
    marker = f"\n{tag}: {insight_text}" if tag else f"\n{insight_text}"
    # Keep at most 3 general strategies in persona to avoid prompt bloat
    import re as _re
    existing = _re.findall(r"\[General strategy\]:.*", agent.profile.persona)
    if len(existing) >= 3:
        # Drop oldest general strategy
        agent.profile.persona = _re.sub(
            r"\n\[General strategy\]:.*", "", agent.profile.persona, count=1
        )
    agent.profile.persona = agent.profile.persona.rstrip() + marker


# Domain relatedness map — skills that belong to the same macro-domain cluster.
# Co-Dream should only update skills within the same cluster as the current task.
_DOMAIN_CLUSTERS = {
    "math": {"math", "gsm8k", "algebra", "calculus", "math_competition", "math_word_problem",
             "arithmetic", "number_theory", "geometry", "precalculus", "combinatorics",
             "counting_and_probability", "prealgebra", "intermediate_algebra"},
    "code": {"code", "mbpp", "humaneval", "programming", "code_completion", "code_generation",
             "coding", "software", "implementation", "recursion", "sorting", "dynamic_programming",
             "string_manipulation", "data_structures", "algorithms", "debugging", "refactoring"},
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
