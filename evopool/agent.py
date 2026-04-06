"""
EvoPool Agent — persistent profile + individual evolution.
Each agent maintains a structured profile that evolves over time.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .llm import llm_call


@dataclass
class AgentProfile:
    """Structured agent profile. This is the evolving identity of an agent."""

    persona: str  # Evolving description of agent's identity, style, specialization
    skill_memory: dict[str, float]  # domain -> confidence score [0, 1]
    task_history: list[dict]  # last N tasks: {type, outcome, score}
    collab_log: dict[str, list[str]]  # agent_id -> [what I learned from them]
    perf_stats: dict[str, list[float]]  # domain -> rolling performance scores

    TASK_HISTORY_LIMIT: int = field(default=20, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = {
            "persona": self.persona,
            "skill_memory": self.skill_memory,
            "task_history": self.task_history[-self.TASK_HISTORY_LIMIT:],
            "collab_log": self.collab_log,
            "perf_stats": {k: v[-20:] for k, v in self.perf_stats.items()},
        }
        # Persist all memory tiers
        if hasattr(self, "subdomain_insights") and self.subdomain_insights:
            d["subdomain_insights"] = self.subdomain_insights
        if hasattr(self, "domain_insights") and self.domain_insights:
            d["domain_insights"] = self.domain_insights  # L2.5
        if hasattr(self, "working_memory") and self.working_memory:
            d["working_memory"] = self.working_memory
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AgentProfile:
        obj = cls(
            persona=d["persona"],
            skill_memory=d["skill_memory"],
            task_history=d["task_history"],
            collab_log=d["collab_log"],
            perf_stats=d["perf_stats"],
        )
        if "subdomain_insights" in d:
            obj.subdomain_insights = d["subdomain_insights"]
        if "domain_insights" in d:
            obj.domain_insights = d["domain_insights"]  # L2.5
        if "working_memory" in d:
            obj.working_memory = d["working_memory"]
        return obj

    def summarize(self) -> str:
        """Short text summary for use in prompts."""
        top_skills = sorted(self.skill_memory.items(), key=lambda x: -x[1])[:5]
        skills_str = ", ".join(f"{k}({v:.2f})" for k, v in top_skills)
        recent_tasks = self.task_history[-5:]
        task_types = [t["type"] for t in recent_tasks]
        hypotheses = getattr(self, "hypotheses", [])
        hyp_str = f"\nActive hypotheses: {hypotheses[:2]}" if hypotheses else ""
        return (
            f"Persona: {self.persona}\n"
            f"Top skills: {skills_str}\n"
            f"Recent task types: {task_types}"
            f"{hyp_str}"
        )

    def affinity_for(self, task_type: str) -> float:
        """Affinity score for a task type (used in selection + leader assignment)."""
        return self.skill_memory.get(task_type, 0.2)

    def dominant_domains(self, top_k: int = 3) -> list[str]:
        """Return the top-k domains by skill confidence."""
        return [k for k, _ in sorted(self.skill_memory.items(), key=lambda x: -x[1])[:top_k]]

    def profile_vector(self) -> list[float]:
        """Return a simple numeric vector over all known domains (for similarity)."""
        domains = sorted(self.skill_memory.keys())
        return [self.skill_memory.get(d, 0.0) for d in domains]


class Agent:
    """
    An EvoPool agent with a persistent evolving profile.
    After each task, calls update_from_feedback() to evolve the profile.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        profile: AgentProfile | None = None,
        parent_ids: list[str] | None = None,
    ):
        self.agent_id: str = agent_id or str(uuid.uuid4())[:8]
        self.profile: AgentProfile = profile or self._init_default_profile()
        self.parent_ids: list[str] = parent_ids or []
        self.created_at: float = time.time()
        self.task_count: int = 0
        self.consecutive_underperformance: int = 0  # for prune trigger
        self._pending_codream: list[dict] = []  # co-dream lessons queued for integration

    def _init_default_profile(self) -> AgentProfile:
        return AgentProfile(
            persona="A general-purpose AI agent with broad capabilities.",
            skill_memory={},
            task_history=[],
            collab_log={},
            perf_stats={},
        )

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Build the agent's system prompt from its profile."""
        return (
            "You are an AI agent with the following profile:\n\n"
            f"{self.profile.summarize()}\n\n"
            "Use your accumulated experience and specializations to complete the task."
        )

    def execute_subtask(
        self,
        task: dict,
        subtask_prompt: str,
        context: str,
        backbone_llm: str,
        max_tokens: int = 512,
    ) -> dict:
        """Execute an assigned subtask with injected context from other agents."""
        system_prompt = self.build_system_prompt()
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        is_code_task = domain in ("mbpp", "humaneval") or task_type in (
            "code_generation", "code_completion"
        )
        is_math_task = domain in ("gsm8k", "math") or task_type in (
            "math_word_problem", "math_competition", "arithmetic"
        )
        is_qa_task = domain in ("hotpotqa", "drop") or task_type in (
            "multi_hop_qa", "reading_comprehension"
        )
        # Only add code format instruction for primary agents, not reviewers
        is_review = "Review the team's work" in subtask_prompt or "identify issues" in subtask_prompt
        # Override system prompt for code tasks: profile persona is too generic for coding
        if is_code_task and not is_review:
            system_prompt = "You are an expert Python programmer. Write clean, correct, and efficient code."
        if is_code_task and not is_review:
            subtask_prompt = (
                subtask_prompt
                + "\n\nIMPORTANT: Output ONLY the complete Python function implementation "
                "in a markdown code block (```python ... ```) with no explanation outside the block. "
                "Use the EXACT function name shown in the test cases or function signature."
            )
        elif is_math_task and not is_review:
            if domain == "gsm8k" or task_type == "math_word_problem":
                subtask_prompt = (
                    subtask_prompt
                    + "\n\nSolve step by step. End your answer with: #### <final number>"
                )
            else:
                subtask_prompt = (
                    subtask_prompt
                    + "\n\nSolve step by step. Before finalizing, verify your answer "
                    "by checking it against the original problem or using an alternative approach. "
                    "Express your final answer in LaTeX inside \\boxed{}. "
                    "For example: \\boxed{\\frac{3}{5}} or \\boxed{42} or \\boxed{x+1}."
                )
        elif is_qa_task and not is_review:
            subtask_prompt = (
                subtask_prompt
                + "\n\nProvide a concise, direct answer. Do not repeat the question."
            )
        # Inject domain_general insights (L2.5): useful for ALL tasks in this broad domain.
        # Higher-priority than subdomain hints; injected for code tasks too (safe meta-strategies).
        if not is_review:
            domain_hint = _get_domain_hint(self.profile, task)
            if domain_hint:
                subtask_prompt = domain_hint + "\n\n" + subtask_prompt

        # Fix 1: inject scoped subdomain insights when sub-domain matches current task.
        # Skip for code tasks (format-sensitive) and reviews.
        if not is_code_task and not is_review:
            subdomain_hint = _get_subdomain_hint(self.profile, task)
            if subdomain_hint:
                subtask_prompt = subdomain_hint + "\n\n" + subtask_prompt

        # Fix 3: inject working memory (task_specific insights from previous task, same domain).
        # These are one-shot hints: used once, then cleared to avoid stale/irrelevant advice.
        if not is_code_task and not is_review:
            wm_hint = _get_working_memory_hint(self.profile, task)
            if wm_hint:
                subtask_prompt = wm_hint + "\n\n" + subtask_prompt

        user_prompt = subtask_prompt
        if context:
            user_prompt = f"Context from teammates:\n{context}\n\n---\nYour task:\n{subtask_prompt}"
        response = llm_call(
            model=backbone_llm,
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
        )
        return {"agent_id": self.agent_id, "response": response, "task_type": task.get("type", "unknown")}

    def execute_task(self, task: dict, backbone_llm: str) -> dict:
        """Execute a task and return result dict (used by self-consistency math/QA path)."""
        system_prompt = self.build_system_prompt()
        user_prompt = task.get("prompt", task.get("question", str(task)))
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        is_code = domain in ("mbpp", "humaneval") or task_type in ("code_generation", "code_completion")
        # Add format hint for self-consistency paths (math only — QA hint hurts reasoning)
        if domain == "gsm8k" or task_type == "math_word_problem":
            user_prompt = user_prompt + "\n\nSolve step by step. End your answer with: #### <final number>"
        elif task_type in ("math_competition", "arithmetic") or domain == "math":
            user_prompt = (
                user_prompt
                + "\n\nSolve step by step. Before finalizing, verify your answer "
                "by checking it against the original problem or using an alternative approach. "
                "Express your final answer in LaTeX inside \\boxed{}. "
                "For example: \\boxed{\\frac{3}{5}} or \\boxed{42} or \\boxed{x+1}."
            )

        # Inject domain_general, subdomain, and working memory hints for math/QA paths.
        domain_hint = _get_domain_hint(self.profile, task)
        if domain_hint:
            user_prompt = domain_hint + "\n\n" + user_prompt
        if not is_code:
            subdomain_hint = _get_subdomain_hint(self.profile, task)
            if subdomain_hint:
                user_prompt = subdomain_hint + "\n\n" + user_prompt
            wm_hint = _get_working_memory_hint(self.profile, task)
            if wm_hint:
                user_prompt = wm_hint + "\n\n" + user_prompt

        # Allocate enough tokens for competition math step-by-step reasoning.
        # QA answers are short so 512 is fine; math needs more headroom.
        is_math = domain in ("gsm8k", "math") or task_type in (
            "math_word_problem", "math_competition", "arithmetic"
        )
        max_tokens = 1024 if is_math else 512

        response = llm_call(
            model=backbone_llm,
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
        )
        return {"agent_id": self.agent_id, "response": response, "task_type": task.get("type", "unknown")}

    # ------------------------------------------------------------------
    # Individual profile evolution (post-task)
    # ------------------------------------------------------------------

    def update_from_feedback(self, task: dict, outcome: dict, backbone_llm: str) -> None:
        """Update profile after a task based on performance feedback."""
        task_type = task.get("type", "general")
        score = outcome.get("score", 0.5)

        # Update skill memory with adaptive learning rate.
        # Use faster updates early (few observations) and slower updates later.
        # Guard: agents with specialist priors (skill > 0.5) treat prior as if
        # they have 2 prior observations, preventing a single failure from
        # collapsing their specialization before they have real experience.
        n_obs = len(self.profile.perf_stats.get(task_type, []))
        prev = self.profile.skill_memory.get(task_type, 0.5)
        # Virtual observations for specialist priors
        if n_obs == 0 and prev > 0.4:
            n_obs_eff = 2  # treat specialist prior as 2 prior observations
        else:
            n_obs_eff = n_obs
        alpha = max(0.1, 0.6 / (1 + n_obs_eff * 0.3))  # starts at 0.42 for specialists
        self.profile.skill_memory[task_type] = (1 - alpha) * prev + alpha * score

        # Update performance stats
        if task_type not in self.profile.perf_stats:
            self.profile.perf_stats[task_type] = []
        self.profile.perf_stats[task_type].append(score)

        # Update task history
        self.profile.task_history.append({
            "type": task_type,
            "outcome": outcome.get("label", "unknown"),
            "score": score,
            "task_id": task.get("id", ""),
        })

        # Update persona (LLM call, lightweight)
        if self.task_count % 5 == 0:  # Only update persona every 5 tasks to save cost
            self._update_persona(task, outcome, backbone_llm)

        self.task_count += 1

        # Track underperformance for prune trigger
        pool_mean = outcome.get("pool_mean_score", 0.5)
        if score < pool_mean * 0.8:
            self.consecutive_underperformance += 1
        else:
            self.consecutive_underperformance = 0

    def _update_persona(self, task: dict, outcome: dict, backbone_llm: str) -> None:
        """Incrementally update the persona string (max 20% semantic drift).

        Extracts and preserves [General strategy] entries added by Co-Dream (Fix 2a)
        before the LLM rewrites the persona — otherwise they would be silently lost
        every 5 tasks since the LLM outputs only a new 1-2 sentence persona.
        """
        import re as _re
        # Extract general strategies to re-append after update
        strategy_entries = _re.findall(r"\n\[General strategy\]:.*", self.profile.persona)
        # Build the base persona (without strategy annotations) for the LLM to revise
        base_persona = _re.sub(r"\n\[General strategy\]:.*", "", self.profile.persona).strip()
        prompt = (
            f"Current agent persona:\n{base_persona}\n\n"
            f"Recent task: type={task.get('type')}, score={outcome.get('score', 0.5):.2f}\n"
            f"Top skills: {self.profile.dominant_domains()}\n\n"
            "Revise the persona (1-2 sentences) to incrementally reflect any emerging "
            "specializations. Keep changes subtle and incremental. Do NOT drastically "
            "change the identity. Output only the revised persona text."
        )
        try:
            new_persona = llm_call(model=backbone_llm, user=prompt, max_tokens=100)
            # Re-attach preserved general strategies so they survive the persona rewrite
            self.profile.persona = new_persona.strip() + "".join(strategy_entries)
        except Exception:
            pass  # Keep current persona on failure

    # ------------------------------------------------------------------
    # Co-Dream integration (called by Co-Dream mechanism)
    # ------------------------------------------------------------------

    def queue_codream_lesson(self, from_agent_id: str, domains: list[str], lesson: str) -> None:
        """Queue a co-dream lesson to be integrated."""
        self._pending_codream.append({
            "from_agent_id": from_agent_id,
            "domains": domains,
            "lesson": lesson,
        })

    def integrate_codream_lessons(self, backbone_llm: str) -> None:
        """Integrate all queued co-dream lessons into the profile."""
        if not self._pending_codream:
            return

        lessons_text = "\n".join(
            f"- From agent {l['from_agent_id']} (their strong domains: {l['domains']}): {l['lesson']}"
            for l in self._pending_codream
        )

        prompt = (
            f"Your current profile:\n{self.profile.summarize()}\n\n"
            f"Lessons learned from collaborators:\n{lessons_text}\n\n"
            "Based on these lessons, update your skill_memory JSON dict. "
            "Only update domains where collaborators demonstrated strength. "
            "Keep changes incremental (max +0.1 per domain). "
            "Output ONLY valid JSON: {\"domain\": score, ...}"
        )

        try:
            response = llm_call(model=backbone_llm, user=prompt, max_tokens=200)
            updates = json.loads(response.strip())
            for domain, new_score in updates.items():
                old = self.profile.skill_memory.get(domain, 0.3)
                # Cap the update to prevent rapid convergence
                self.profile.skill_memory[domain] = min(old + 0.1, float(new_score))
        except Exception:
            pass  # Fail silently; keep existing profile

        # Update collab log
        for lesson in self._pending_codream:
            fid = lesson["from_agent_id"]
            if fid not in self.profile.collab_log:
                self.profile.collab_log[fid] = []
            self.profile.collab_log[fid].append(lesson["lesson"][:200])

        self._pending_codream.clear()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "profile": self.profile.to_dict(),
            "parent_ids": self.parent_ids,
            "created_at": self.created_at,
            "task_count": self.task_count,
            "consecutive_underperformance": self.consecutive_underperformance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Agent:
        agent = cls(
            agent_id=d["agent_id"],
            profile=AgentProfile.from_dict(d["profile"]),
            parent_ids=d.get("parent_ids", []),
        )
        agent.created_at = d.get("created_at", time.time())
        agent.task_count = d.get("task_count", 0)
        agent.consecutive_underperformance = d.get("consecutive_underperformance", 0)
        return agent


# ------------------------------------------------------------------
# Fix 2 / Fix 3: Subdomain + working memory injection helpers
# ------------------------------------------------------------------

def _get_domain_hint(profile: "AgentProfile", task: dict) -> str:
    """
    Inject domain_general insights (L2.5): strategies useful for ALL tasks in the
    same broad domain (e.g. all MATH, all code). These are higher priority than
    subdomain hints and apply regardless of subtopic.
    """
    domain_insights: dict = getattr(profile, 'domain_insights', {})
    if not domain_insights:
        return ""

    from .codream import _domains_related, _DOMAIN_CLUSTERS
    task_domain = task.get("domain", task.get("type", ""))

    matched: list[str] = []
    for scope, insights in domain_insights.items():
        if _domains_related(scope, task_domain):
            matched.extend(insights)

    if not matched:
        return ""

    hint = "[Broad domain strategy from past experience — apply when relevant]:\n"
    hint += "\n".join(f"- {ins}" for ins in matched[:2])
    return hint


def _get_subdomain_hint(profile: "AgentProfile", task: dict) -> str:
    """
    Look up subdomain_insights stored on this agent's profile and return a
    formatted hint string if any scoped insights match the current task's domain.

    Uses _domains_related() from codream.py to check if a stored scope (e.g. "algebra")
    is in the same cluster as the current task's domain (e.g. "math").
    This is more robust than keyword matching and reuses the existing cluster logic.

    Only called for non-code, non-review tasks (Fix 1 injection).
    """
    subdomain_insights: dict = getattr(profile, 'subdomain_insights', {})
    if not subdomain_insights:
        return ""

    from .codream import _domains_related
    task_domain = task.get("domain", task.get("type", ""))

    matched_insights: list[str] = []
    for scope, insights in subdomain_insights.items():
        # Only inject if the scope belongs to the same domain cluster as the current task.
        # e.g. algebra + math → True; algebra + drop → False; recursion + humaneval → True
        if _domains_related(scope, task_domain):
            matched_insights.extend(insights)

    if not matched_insights:
        return ""

    hint = "[Scoped strategy from past experience — apply only if relevant to this sub-topic]:\n"
    hint += "\n".join(f"- {ins}" for ins in matched_insights[:2])  # max 2 to avoid bloat
    return hint


def _get_working_memory_hint(profile: "AgentProfile", task: dict) -> str:
    """
    Fix 3: Inject and clear working memory (task_specific insights from the previous task).

    working_memory stores [(domain, insight), ...] tuples written by _apply_insights
    when the Co-Dream produces a task_specific insight.  We inject those that belong to
    the same macro-domain cluster as the current task, then CLEAR the entire working
    memory so stale hints do not persist across domain switches.

    One-shot: the hint is used exactly once (on the very next task), then discarded.
    """
    working_mem: list[tuple[str, str]] = getattr(profile, 'working_memory', [])
    if not working_mem:
        return ""

    from .codream import _domains_related
    task_domain = task.get("domain", task.get("type", ""))

    relevant = [ins for (dom, ins) in working_mem if _domains_related(dom, task_domain)]

    # Always clear after reading — working memory is one-shot regardless of domain match
    profile.working_memory = []

    if not relevant:
        return ""

    hint = "[Lesson from last similar task — apply only if directly relevant]:\n"
    hint += "\n".join(f"- {ins}" for ins in relevant[:2])
    return hint

