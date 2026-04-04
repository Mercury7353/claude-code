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
        return {
            "persona": self.persona,
            "skill_memory": self.skill_memory,
            "task_history": self.task_history[-self.TASK_HISTORY_LIMIT:],
            "collab_log": self.collab_log,
            "perf_stats": {k: v[-20:] for k, v in self.perf_stats.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> AgentProfile:
        return cls(
            persona=d["persona"],
            skill_memory=d["skill_memory"],
            task_history=d["task_history"],
            collab_log=d["collab_log"],
            perf_stats=d["perf_stats"],
        )

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
                    + "\n\nSolve step by step. Box your final answer: \\boxed{<answer>}"
                )
        elif is_qa_task and not is_review:
            subtask_prompt = (
                subtask_prompt
                + "\n\nProvide a concise, direct answer. Do not repeat the question."
            )
        user_prompt = subtask_prompt
        if context:
            user_prompt = f"Context from teammates:\n{context}\n\n---\nYour task:\n{subtask_prompt}"
        response = llm_call(
            model=backbone_llm,
            system=system_prompt,
            user=user_prompt,
        )
        return {"agent_id": self.agent_id, "response": response, "task_type": task.get("type", "unknown")}

    def execute_task(self, task: dict, backbone_llm: str) -> dict:
        """Execute a task and return result dict (used by self-consistency math path)."""
        system_prompt = self.build_system_prompt()
        user_prompt = task.get("prompt", task.get("question", str(task)))
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        # Add format hint for math self-consistency
        if domain == "gsm8k" or task_type == "math_word_problem":
            user_prompt = user_prompt + "\n\nSolve step by step. End your answer with: #### <final number>"
        elif task_type in ("math_competition", "arithmetic") or domain == "math":
            user_prompt = user_prompt + "\n\nSolve step by step. Box your final answer: \\boxed{<answer>}"

        response = llm_call(
            model=backbone_llm,
            system=system_prompt,
            user=user_prompt,
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
        # Use faster updates early (few observations) and slower updates later
        # to prevent rapid convergence and preserve specialization.
        n_obs = len(self.profile.perf_stats.get(task_type, []))
        alpha = max(0.1, 0.6 / (1 + n_obs * 0.3))  # starts at 0.6, decays toward 0.1
        prev = self.profile.skill_memory.get(task_type, 0.5)
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
        """Incrementally update the persona string (max 20% semantic drift)."""
        prompt = (
            f"Current agent persona:\n{self.profile.persona}\n\n"
            f"Recent task: type={task.get('type')}, score={outcome.get('score', 0.5):.2f}\n"
            f"Top skills: {self.profile.dominant_domains()}\n\n"
            "Revise the persona (1-2 sentences) to incrementally reflect any emerging "
            "specializations. Keep changes subtle and incremental. Do NOT drastically "
            "change the identity. Output only the revised persona text."
        )
        try:
            new_persona = llm_call(model=backbone_llm, user=prompt, max_tokens=100)
            self.profile.persona = new_persona.strip()
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
