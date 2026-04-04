"""
Dynamic MAS with Team Leader.

Execution flow per task:
  1. Leader is selected (agent with highest affinity for the task type)
  2. Leader analyzes task → determines complexity + required skills
  3. Leader decomposes task → assigns subtasks to each team member
     - Simple task: one primary agent + reviewer(s)
     - Complex task: named subtasks matched to agent strengths
     - GenesisOnDemand: if no team member covers a required skill, recruit from full pool
  4. Round 1: agents execute their assigned subtasks (parallel)
  5. Round 2 (optional): reviewer agents critique Round 1 output
  6. Leader synthesizes all results → final answer
  7. Leader generates per-agent feedback (process quality signal)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .llm import llm_call

if TYPE_CHECKING:
    from .agent import Agent

SKILL_FLOOR = 0.35  # If best team agent has skill < this, trigger GenesisOnDemand


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class SubtaskAssignment:
    agent_id: str
    role: str               # "primary" | "reviewer" | "subtask:<name>"
    subtask_prompt: str
    context_from_others: str = ""
    required_skills: list[str] = field(default_factory=list)


@dataclass
class SubtaskResult:
    agent_id: str
    role: str
    subtask_name: str
    response: str


@dataclass
class MASResult:
    final_answer: str
    leader_id: str
    per_agent_responses: dict[str, dict]    # agent_id -> {response, task_type, role}
    per_agent_feedback: dict[str, dict]     # agent_id -> {score_hint, feedback_text}
    decomposition_plan: list[dict]
    extra_agents_recruited: list[str]
    n_rounds: int


# ------------------------------------------------------------------
# TeamLeader
# ------------------------------------------------------------------

class TeamLeader:
    """
    Orchestrates task execution for a team of agents.
    The leader itself is an Agent (the one with highest task affinity).
    """

    def __init__(
        self,
        leader: Agent,
        team: list[Agent],
        pool: list[Agent],
        backbone_llm: str,
        max_extra_agents: int = 2,
        critique_enabled: bool = True,
    ):
        self.leader = leader
        # Team includes the leader
        self.team = team
        self.pool = pool
        self.backbone_llm = backbone_llm
        self.max_extra_agents = max_extra_agents
        self.critique_enabled = critique_enabled

    def run(self, task: dict) -> MASResult:
        """Full leader-coordinated task execution pipeline."""
        extra_recruited: list[str] = []

        # Step 1: Analyze task
        analysis = self._analyze_task(task)

        # Step 2: Decompose into subtask assignments
        assignments, newly_recruited = self._decompose_task(task, analysis)
        extra_recruited.extend(newly_recruited)

        # Step 3: Round 1 — execute all subtasks
        results_r1 = self._execute_assignments(assignments, task)

        # Step 4: Round 2 — critique (optional)
        results_r2 = []
        if self.critique_enabled:
            results_r2 = self._critique_round(results_r1, task)

        all_results = results_r1 + results_r2

        # Step 5: Synthesize
        final_answer = self._synthesize(task, all_results)

        # Step 6: Per-agent feedback
        feedback = self._generate_feedback(task, all_results)

        # Build responses dict (same shape as flat-mode execute_task output)
        per_agent_responses: dict[str, dict] = {}
        for r in all_results:
            if r.agent_id not in per_agent_responses:
                per_agent_responses[r.agent_id] = {
                    "agent_id": r.agent_id,
                    "response": r.response,
                    "task_type": task.get("type", "unknown"),
                    "role": r.role,
                    "subtask": r.subtask_name,
                }
        # Also add the final synthesized answer under the leader's entry
        if self.leader.agent_id in per_agent_responses:
            per_agent_responses[self.leader.agent_id]["synthesized_answer"] = final_answer

        return MASResult(
            final_answer=final_answer,
            leader_id=self.leader.agent_id,
            per_agent_responses=per_agent_responses,
            per_agent_feedback=feedback,
            decomposition_plan=[
                {"agent_id": a.agent_id, "role": a.role, "skills": a.required_skills}
                for a in assignments
            ],
            extra_agents_recruited=extra_recruited,
            n_rounds=1 + (1 if results_r2 else 0),
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _analyze_task(self, task: dict) -> dict:
        """Leader analyzes the task and determines complexity + skill requirements."""
        team_profiles = "\n".join(
            f"- Agent {a.agent_id}: {a.profile.summarize()}"
            for a in self.team
        )
        prompt = (
            f"You are the team leader analyzing a task to coordinate your team.\n\n"
            f"Task type: {task.get('type', 'general')}\n"
            f"Task: {task.get('prompt', '')[:500]}\n\n"
            f"Your team members:\n{team_profiles}\n\n"
            "Analyze this task and respond with a JSON object:\n"
            "{\n"
            '  "complexity": "simple" or "complex",\n'
            '  "required_skills": ["skill1", "skill2", ...],\n'
            '  "subtask_names": ["subtask1", "subtask2", ...],\n'
            '  "reasoning": "brief explanation"\n'
            "}\n\n"
            "Simple = one agent can solve it alone with others reviewing.\n"
            "Complex = genuinely requires multiple distinct contributions.\n"
            "Output ONLY valid JSON."
        )
        try:
            raw = llm_call(
                model=self.backbone_llm,
                system=self.leader.build_system_prompt(),
                user=prompt,
                max_tokens=300,
            )
            return json.loads(raw.strip())
        except Exception:
            return {
                "complexity": "simple",
                "required_skills": [task.get("type", "general")],
                "subtask_names": [],
            }

    def _decompose_task(
        self, task: dict, analysis: dict
    ) -> tuple[list[SubtaskAssignment], list[str]]:
        """Build SubtaskAssignment list. Returns (assignments, newly_recruited_agent_ids)."""
        complexity = analysis.get("complexity", "simple")
        required_skills = analysis.get("required_skills", [task.get("type", "general")])
        subtask_names = analysis.get("subtask_names", [])
        recruited: list[str] = []

        working_team = list(self.team)

        if complexity == "simple" or len(subtask_names) <= 1:
            return self._simple_decompose(task, working_team), recruited

        # Complex: assign named subtasks
        assignments: list[SubtaskAssignment] = []
        used_agent_ids: set[str] = set()

        for skill, subtask_name in zip(required_skills, subtask_names):
            best = self._best_agent_for_skill(skill, working_team, exclude=used_agent_ids)

            if best is None or best.profile.skill_memory.get(skill, 0.2) < SKILL_FLOOR:
                # GenesisOnDemand: recruit from full pool
                candidate = self._recruit_from_pool(skill, used_agent_ids)
                if candidate and len(recruited) < self.max_extra_agents:
                    working_team.append(candidate)
                    recruited.append(candidate.agent_id)
                    best = candidate

            if best is None:
                best = working_team[0]  # fallback

            used_agent_ids.add(best.agent_id)
            assignments.append(SubtaskAssignment(
                agent_id=best.agent_id,
                role=f"subtask:{subtask_name}",
                subtask_prompt=self._build_subtask_prompt(task, subtask_name, best),
                required_skills=[skill],
            ))

        # Any unassigned team members become reviewers
        for agent in working_team:
            if agent.agent_id not in used_agent_ids:
                assignments.append(SubtaskAssignment(
                    agent_id=agent.agent_id,
                    role="reviewer",
                    subtask_prompt=self._build_review_prompt(task),
                    required_skills=[],
                ))

        return assignments, recruited

    def _simple_decompose(self, task: dict, team: list[Agent]) -> list[SubtaskAssignment]:
        """Simple mode: one primary agent solves, others review."""
        task_type = task.get("type", "general")
        primary = max(team, key=lambda a: a.profile.skill_memory.get(task_type, 0.2))
        assignments = [SubtaskAssignment(
            agent_id=primary.agent_id,
            role="primary",
            subtask_prompt=task.get("prompt", str(task)),
            required_skills=[task_type],
        )]
        for agent in team:
            if agent.agent_id != primary.agent_id:
                assignments.append(SubtaskAssignment(
                    agent_id=agent.agent_id,
                    role="reviewer",
                    subtask_prompt=self._build_review_prompt(task),
                    required_skills=[],
                ))
        return assignments

    def _execute_assignments(
        self, assignments: list[SubtaskAssignment], task: dict
    ) -> list[SubtaskResult]:
        """Execute all subtask assignments (Round 1). Returns results in assignment order."""
        agent_map = {a.agent_id: a for a in self.team + self._extra_agents()}
        results: list[SubtaskResult] = []
        accumulated_context = ""

        for assignment in assignments:
            agent = agent_map.get(assignment.agent_id)
            if agent is None:
                continue

            # Inject accumulated context from prior subtasks
            context = accumulated_context
            response = agent.execute_subtask(
                task=task,
                subtask_prompt=assignment.subtask_prompt,
                context=context,
                backbone_llm=self.backbone_llm,
            )
            result = SubtaskResult(
                agent_id=assignment.agent_id,
                role=assignment.role,
                subtask_name=assignment.role,
                response=response.get("response", ""),
            )
            results.append(result)
            # Accumulate context for subsequent agents
            accumulated_context += f"\n[{assignment.role} by {assignment.agent_id}]: {result.response[:300]}"

        return results

    def _critique_round(
        self, results_r1: list[SubtaskResult], task: dict
    ) -> list[SubtaskResult]:
        """Round 2: reviewer agents provide critique/improvement of Round 1."""
        agent_map = {a.agent_id: a for a in self.team + self._extra_agents()}
        r1_summary = "\n".join(
            f"[{r.role}]: {r.response[:400]}" for r in results_r1
        )
        critique_results: list[SubtaskResult] = []

        for r in results_r1:
            if r.role != "reviewer":
                continue
            agent = agent_map.get(r.agent_id)
            if agent is None:
                continue

            critique_prompt = (
                f"Original task: {task.get('prompt', '')[:400]}\n\n"
                f"Team's work so far:\n{r1_summary}\n\n"
                "As a reviewer, identify the most important improvement or correction. "
                "Be specific and constructive. Focus on correctness over style."
            )
            response = agent.execute_subtask(
                task=task,
                subtask_prompt=critique_prompt,
                context="",
                backbone_llm=self.backbone_llm,
            )
            critique_results.append(SubtaskResult(
                agent_id=r.agent_id,
                role="critique",
                subtask_name="critique",
                response=response.get("response", ""),
            ))

        return critique_results

    def _synthesize(self, task: dict, all_results: list[SubtaskResult]) -> str:
        """Leader synthesizes all subtask results into a final answer."""
        results_text = "\n\n".join(
            f"=== {r.role} (Agent {r.agent_id}) ===\n{r.response}"
            for r in all_results
        )
        domain = task.get("domain", "")
        is_code_task = domain in ("mbpp", "humaneval") or task.get("type", "") in ("code_generation", "code_completion")
        if is_code_task:
            format_instruction = (
                "Output ONLY the Python function implementation in a markdown code block. "
                "Format: ```python\n<your code here>\n``` "
                "Do NOT include explanations, test cases, or any text outside the code block."
            )
        else:
            format_instruction = (
                "Incorporate the best contributions and address any critiques. "
                "Output only the final answer."
            )
        prompt = (
            f"Task: {task.get('prompt', '')[:600]}\n\n"
            f"Your team's work:\n{results_text}\n\n"
            f"Synthesize the above into a single, complete, correct final answer. "
            f"{format_instruction}"
        )
        try:
            return llm_call(
                model=self.backbone_llm,
                system=self.leader.build_system_prompt(),
                user=prompt,
                max_tokens=800,
            )
        except Exception:
            # Fallback: return the primary agent's response
            for r in all_results:
                if r.role == "primary":
                    return r.response
            return all_results[0].response if all_results else ""

    def _generate_feedback(
        self, task: dict, all_results: list[SubtaskResult]
    ) -> dict[str, dict]:
        """Leader assesses each agent's contribution quality."""
        results_text = "\n".join(
            f"Agent {r.agent_id} ({r.role}): {r.response[:200]}"
            for r in all_results
        )
        prompt = (
            f"Task type: {task.get('type', 'general')}\n"
            f"Agent contributions:\n{results_text}\n\n"
            "Rate each agent's contribution quality from 0.0 to 1.0.\n"
            "Output JSON: {\"agent_id\": {\"score_hint\": float, \"feedback\": str}, ...}"
        )
        try:
            raw = llm_call(
                model=self.backbone_llm,
                system="You are an objective evaluator of agent performance.",
                user=prompt,
                max_tokens=400,
            )
            return json.loads(raw.strip())
        except Exception:
            return {r.agent_id: {"score_hint": 0.5, "feedback": ""} for r in all_results}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_agent_for_skill(
        self, skill: str, team: list[Agent], exclude: set[str]
    ) -> Agent | None:
        candidates = [a for a in team if a.agent_id not in exclude]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.profile.skill_memory.get(skill, 0.2))

    def _recruit_from_pool(self, skill: str, exclude: set[str]) -> Agent | None:
        candidates = [a for a in self.pool if a.agent_id not in exclude]
        if not candidates:
            return None
        best = max(candidates, key=lambda a: a.profile.skill_memory.get(skill, 0.2))
        return best if best.profile.skill_memory.get(skill, 0.2) > SKILL_FLOOR else None

    def _extra_agents(self) -> list[Agent]:
        """Extra recruited agents not in original team."""
        team_ids = {a.agent_id for a in self.team}
        return [a for a in self.pool if a.agent_id not in team_ids]

    def _build_subtask_prompt(self, task: dict, subtask_name: str, agent: Agent) -> str:
        return (
            f"Original task: {task.get('prompt', '')[:500]}\n\n"
            f"Your assigned subtask: {subtask_name}\n"
            f"Your specialization: {', '.join(agent.profile.dominant_domains())}\n\n"
            "Complete your assigned subtask thoroughly."
        )

    def _build_review_prompt(self, task: dict) -> str:
        return (
            f"Original task: {task.get('prompt', '')[:500]}\n\n"
            "Review the team's work and identify issues, errors, or improvements."
        )


# ------------------------------------------------------------------
# Module-level entry point (called from pool.py)
# ------------------------------------------------------------------

def run_leader_mas(
    team: list[Agent],
    task: dict,
    pool: list[Agent],
    backbone_llm: str,
    max_extra_agents: int = 2,
    critique_enabled: bool = True,
) -> MASResult:
    """Select a leader and run the full leader-coordinated task execution."""
    task_type = task.get("type", "general")
    leader = max(team, key=lambda a: a.profile.skill_memory.get(task_type, 0.2))

    coordinator = TeamLeader(
        leader=leader,
        team=team,
        pool=pool,
        backbone_llm=backbone_llm,
        max_extra_agents=max_extra_agents,
        critique_enabled=critique_enabled,
    )
    return coordinator.run(task)
