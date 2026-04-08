"""
EvoPool Agent — persistent profile + individual evolution.
Each agent maintains a structured profile that evolves over time.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .llm import llm_call


@dataclass
class Experience:
    """A concrete solving experience stored in an agent's private buffer."""
    task_id: str
    domain: str              # e.g. "aime_2022", "gsm8k", "math_hard"
    task_type: str           # e.g. "math_competition_hard", "multi_hop_qa"
    score: float             # 0.0 or 1.0
    strategy_summary: str    # 1-2 sentences: what approach was used
    lesson: str              # 1 sentence: what to do/avoid next time
    source: str = "self"     # "self" | "codream:<agent_id>"
    relevance_weight: float = 1.0  # decays if unhelpful, grows if useful


@dataclass
class AgentProfile:
    """Structured agent profile. This is the evolving identity of an agent."""

    _DOMAIN_CATEGORY_CACHE: ClassVar[dict[str, str]] = {}

    persona: str  # Evolving description of agent's identity, style, specialization
    skill_memory: dict[str, float]  # domain -> confidence score [0, 1]
    task_history: list[dict]  # last N tasks: {type, outcome, score}
    collab_log: dict[str, list[str]]  # agent_id -> [what I learned from them]
    perf_stats: dict[str, list[float]]  # domain -> rolling performance scores
    experience_buffer: list[Experience] = field(default_factory=list)  # private experiences

    TASK_HISTORY_LIMIT: int = field(default=20, repr=False, compare=False)
    EXPERIENCE_BUFFER_LIMIT: int = field(default=50, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = {
            "persona": self.persona,
            "skill_memory": self.skill_memory,
            "task_history": self.task_history[-self.TASK_HISTORY_LIMIT:],
            "collab_log": self.collab_log,
            "perf_stats": {k: v[-20:] for k, v in self.perf_stats.items()},
            "experience_buffer": [
                {"task_id": e.task_id, "domain": e.domain, "task_type": e.task_type,
                 "score": e.score, "strategy_summary": e.strategy_summary,
                 "lesson": e.lesson, "source": e.source,
                 "relevance_weight": e.relevance_weight}
                for e in self.experience_buffer[-self.EXPERIENCE_BUFFER_LIMIT:]
            ],
        }
        # Persist all memory tiers (legacy — kept for backward compat)
        if hasattr(self, "subdomain_insights") and self.subdomain_insights:
            d["subdomain_insights"] = self.subdomain_insights
        if hasattr(self, "domain_insights") and self.domain_insights:
            d["domain_insights"] = self.domain_insights
        if hasattr(self, "working_memory") and self.working_memory:
            d["working_memory"] = self.working_memory
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AgentProfile:
        exp_buf = [
            Experience(**e) for e in d.get("experience_buffer", [])
        ]
        obj = cls(
            persona=d["persona"],
            skill_memory=d["skill_memory"],
            task_history=d["task_history"],
            collab_log=d["collab_log"],
            perf_stats=d["perf_stats"],
            experience_buffer=exp_buf,
        )
        if "subdomain_insights" in d:
            obj.subdomain_insights = d["subdomain_insights"]
        if "domain_insights" in d:
            obj.domain_insights = d["domain_insights"]
        if "working_memory" in d:
            obj.working_memory = d["working_memory"]
        return obj

    def summarize(self) -> str:
        """Short text summary for use in prompts."""
        top_skills = sorted(self.skill_memory.items(), key=lambda x: -x[1])[:5]
        skills_str = ", ".join(f"{k}({v:.2f})" for k, v in top_skills)
        recent_tasks = self.task_history[-5:]
        task_types = [t["type"] for t in recent_tasks]
        return (
            f"Persona: {self.persona}\n"
            f"Top skills: {skills_str}\n"
            f"Recent task types: {task_types}"
        )

    def compute_persona(self) -> str:
        """Generate persona from actual performance data (replaces LLM rewrite)."""
        domain_stats: dict[str, tuple[int, int]] = {}
        for exp in self.experience_buffer:
            d = exp.domain
            c, t = domain_stats.get(d, (0, 0))
            domain_stats[d] = (c + int(exp.score >= 0.5), t + 1)

        strong = [(d, c / t) for d, (c, t) in domain_stats.items() if t >= 3 and c / t >= 0.6]
        weak = [(d, c / t) for d, (c, t) in domain_stats.items() if t >= 3 and c / t < 0.4]

        # Top strategies from high-weight successful experiences
        good_exps = sorted(
            [e for e in self.experience_buffer if e.score >= 0.5],
            key=lambda e: -e.relevance_weight,
        )[:3]

        parts = []
        if strong:
            parts.append("Strong: " + ", ".join(f"{d}({r:.0%})" for d, r in strong))
        if weak:
            parts.append("Weak: " + ", ".join(f"{d}({r:.0%})" for d, r in weak))
        if good_exps:
            parts.append("Strategies: " + "; ".join(e.strategy_summary[:60] for e in good_exps))
        return ". ".join(parts) if parts else self.persona

    @staticmethod
    def _domain_category(domain: str) -> str:
        """Map a specific domain to its broader category for cross-domain matching.

        Without this, math_hard experiences can't transfer to AIME tasks
        because neither domain nor task_type match.
        """
        if domain in AgentProfile._DOMAIN_CATEGORY_CACHE:
            return AgentProfile._DOMAIN_CATEGORY_CACHE[domain]
        # Math domains
        if domain in ("math_hard", "math", "gsm8k") or domain.startswith("aime_"):
            cat = "math"
        # Code domains
        elif domain in ("mbpp", "humaneval", "code_contests"):
            cat = "code"
        else:
            cat = domain
        AgentProfile._DOMAIN_CATEGORY_CACHE[domain] = cat
        return cat

    def get_relevant_experiences(self, task: dict, max_k: int = 3) -> list[Experience]:
        """Retrieve top-k relevant experiences for a task.

        Matches by: exact domain > same category > same task_type.
        This allows math_hard experiences to transfer to AIME tasks.
        """
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        category = self._domain_category(domain)

        scored: list[tuple[float, Experience]] = []
        for e in self.experience_buffer:
            if e.domain == domain:
                # Exact domain match (highest priority)
                priority = 2.0
            elif self._domain_category(e.domain) == category:
                # Same category (e.g., math_hard → aime)
                priority = 1.0
            elif e.task_type == task_type:
                # Same task type
                priority = 0.5
            else:
                continue
            scored.append((priority + e.relevance_weight, e))

        if not scored:
            return []
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:max_k]]

    def format_experience_hint(self, task: dict) -> str:
        """Format relevant experiences as a short user-prompt injection (<100 tokens)."""
        exps = self.get_relevant_experiences(task, max_k=3)
        if not exps:
            return ""
        lines = ["[Past experience on similar problems]:"]
        for e in exps:
            mark = "✓" if e.score >= 0.5 else "✗"
            lines.append(f"- {mark} \"{e.strategy_summary}\" ({e.task_id}, {'solved' if e.score >= 0.5 else 'failed'})")
        return "\n".join(lines)

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
        """Build system prompt. Kept minimal — private context comes via user prompt."""
        return "You are a helpful AI assistant."

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
        # Skip for code tasks — format-sensitive, domain hints add noise to code generation.
        if not is_code_task and not is_review:
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

        is_hard_math = task.get("type") in ("aime_problem", "math_competition_hard") or task.get("domain", "").startswith("aime_") or task.get("domain") == "math_hard"
        user_prompt = subtask_prompt
        if context:
            user_prompt = f"Context from teammates:\n{context}\n\n---\nYour task:\n{subtask_prompt}"
        effective_max_tokens = max(max_tokens, 7000) if is_hard_math else max_tokens
        response = llm_call(
            model=backbone_llm,
            system=system_prompt,
            user=user_prompt,
            max_tokens=effective_max_tokens,
            enable_thinking=is_hard_math,
        )
        # Retry with thinking disabled if response was truncated (no \boxed{})
        if is_hard_math and "\\boxed" not in response and "<think>" in response:
            response = llm_call(
                model=backbone_llm,
                system=system_prompt,
                user=user_prompt + "\n\nBe concise. Go directly to the answer.",
                max_tokens=effective_max_tokens,
                enable_thinking=False,
            )
        return {"agent_id": self.agent_id, "response": response, "task_type": task.get("type", "unknown")}

    def execute_task(self, task: dict, backbone_llm: str, temperature: float = 0.7,
                     leader_hint: str = "") -> dict:
        """Execute a task independently. Returns response dict.

        Args:
            leader_hint: optional per-agent guidance from the team leader (injected into prompt).
        """
        system_prompt = self.build_system_prompt()
        user_prompt = task.get("prompt", task.get("question", str(task)))
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        is_code = domain in ("mbpp", "humaneval") or task_type in ("code_generation", "code_completion")
        is_hard_math = (
            task_type in ("aime_problem", "math_competition_hard")
            or domain.startswith("aime_") or domain == "math_hard"
        )

        # Format hints by task type
        if domain == "gsm8k" or task_type == "math_word_problem":
            user_prompt += "\n\nSolve step by step. End your answer with: #### <final number>"
        elif task_type in ("math_competition", "math_competition_hard", "arithmetic") or domain in ("math", "math_hard"):
            user_prompt += (
                "\n\nSolve step by step. Verify your answer by checking it against the "
                "original problem. Express your final answer in LaTeX inside \\boxed{}."
            )
        elif task_type == "aime_problem" or domain.startswith("aime_"):
            user_prompt += (
                "\n\nSolve step by step. AIME answers are integers from 000 to 999. "
                "State your final answer as: The answer is: [integer]"
            )

        # Inject private experience (concrete, lightweight, ≤100 tokens)
        exp_hint = self.profile.format_experience_hint(task)
        if exp_hint:
            user_prompt = exp_hint + "\n\n" + user_prompt

        # Inject leader's per-agent guidance if provided
        if leader_hint:
            user_prompt = f"[Team leader guidance]: {leader_hint}\n\n" + user_prompt

        # Token budgets
        if is_hard_math:
            max_tokens = 7000
        elif domain in ("gsm8k", "math") or task_type in ("math_word_problem", "math_competition", "arithmetic"):
            max_tokens = 1024
        else:
            max_tokens = 512

        response = llm_call(
            model=backbone_llm,
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=is_hard_math,
        )
        # Retry with thinking disabled if response was truncated (no \boxed{} or answer)
        if is_hard_math and "\\boxed" not in response and "answer is" not in response.lower() and "<think>" in response:
            response = llm_call(
                model=backbone_llm,
                system=system_prompt,
                user=user_prompt + "\n\nBe concise. Go directly to the answer.",
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=False,
            )
        return {"agent_id": self.agent_id, "response": response, "task_type": task_type}

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

        self.task_count += 1

        # Track underperformance for prune trigger
        pool_mean = outcome.get("pool_mean_score", 0.5)
        if score < pool_mean * 0.8:
            self.consecutive_underperformance += 1
        else:
            self.consecutive_underperformance = 0

    def generate_experience(self, task: dict, response: str, score: float,
                            backbone_llm: str) -> None:
        """After evaluation, generate a concrete experience entry (1 LLM call)."""
        task_id = task.get("id", "unknown")
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        prompt_snippet = task.get("prompt", task.get("question", ""))[:200]

        if score >= 0.5:
            gen_prompt = (
                f"You just solved a {task_type} problem correctly.\n"
                f"Problem: {prompt_snippet}...\n"
                f"Your response (first 300 chars): {response[:300]}...\n\n"
                "In 1-2 short sentences each, provide:\n"
                '{"strategy": "what approach/method you used", "lesson": "key takeaway for similar problems"}'
            )
        else:
            gen_prompt = (
                f"You failed a {task_type} problem (score={score:.1f}).\n"
                f"Problem: {prompt_snippet}...\n"
                f"Your response (first 300 chars): {response[:300]}...\n\n"
                "In 1-2 short sentences each, provide:\n"
                '{"strategy": "what approach you tried", "lesson": "what to do differently next time"}'
            )
        try:
            raw = llm_call(model=backbone_llm, user=gen_prompt, max_tokens=150)
            data = json.loads(raw.strip()) if "{" in raw else {"strategy": raw[:80], "lesson": ""}
            exp = Experience(
                task_id=task_id, domain=domain, task_type=task_type, score=score,
                strategy_summary=data.get("strategy", "")[:100],
                lesson=data.get("lesson", "")[:100],
                source="self",
            )
            self.profile.experience_buffer.append(exp)
            # Cap buffer per domain (keep most recent + highest weight)
            self._trim_experience_buffer()
        except Exception:
            # Fallback: store minimal experience without LLM
            self.profile.experience_buffer.append(Experience(
                task_id=task_id, domain=domain, task_type=task_type, score=score,
                strategy_summary="(no summary)", lesson="(no lesson)", source="self",
            ))

    def _trim_experience_buffer(self) -> None:
        """Keep buffer within limit, prioritizing recent + high-weight entries."""
        buf = self.profile.experience_buffer
        if len(buf) <= self.profile.EXPERIENCE_BUFFER_LIMIT:
            return
        # Sort by relevance_weight desc, then recency (index)
        indexed = [(i, e) for i, e in enumerate(buf)]
        indexed.sort(key=lambda x: (-x[1].relevance_weight, -x[0]))
        keep_indices = set(i for i, _ in indexed[:self.profile.EXPERIENCE_BUFFER_LIMIT])
        self.profile.experience_buffer = [e for i, e in enumerate(buf) if i in keep_indices]

    def update_experience_weights(self, task: dict, score: float) -> None:
        """After a task, adjust relevance_weight of experiences that were injected.

        Natural verification: useful strategies survive, useless ones decay.
        """
        relevant = self.profile.get_relevant_experiences(task, max_k=3)
        for exp in relevant:
            if score >= 0.5:
                exp.relevance_weight = min(2.0, exp.relevance_weight + 0.1)
            else:
                exp.relevance_weight = max(0.0, exp.relevance_weight - 0.2)
        # Prune dead-weight experiences
        self.profile.experience_buffer = [
            e for e in self.profile.experience_buffer if e.relevance_weight > 0.1
        ]

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

