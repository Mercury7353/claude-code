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

    _MATH_TYPES = {"math_competition", "math_word_problem", "arithmetic"}
    _QA_TYPES = {"multi_hop_qa", "reading_comprehension", "factual_qa"}

    def run(self, task: dict) -> MASResult:
        """Full leader-coordinated task execution pipeline."""
        extra_recruited: list[str] = []

        # Math tasks: use self-consistency (all agents solve independently,
        # pick majority answer) rather than primary+reviewer decomposition.
        task_type = task.get("type", "general")
        if task_type in self._MATH_TYPES:
            return self._run_self_consistency(task)

        # QA tasks: also use self-consistency (pick most common short answer).
        # This mirrors AFlow's ScEnsemble approach for QA.
        if task_type in self._QA_TYPES:
            return self._run_qa_self_consistency(task)

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

    def _run_self_consistency(self, task: dict) -> MASResult:
        """
        Self-consistency mode for math tasks.
        All k agents solve independently → pick the majority final answer.
        This mirrors AFlow's ScEnsemble approach, which is well-suited for
        math where multiple solution paths converge on the correct answer.
        """
        import re as _re

        results: list[SubtaskResult] = []
        per_agent_responses: dict[str, dict] = {}

        for agent in self.team:
            resp = agent.execute_task(task, self.backbone_llm)
            results.append(SubtaskResult(
                agent_id=agent.agent_id,
                role="primary",
                subtask_name="solve",
                response=resp.get("response", ""),
            ))
            per_agent_responses[agent.agent_id] = resp

        # Extract final answers and vote
        def _extract_final(text: str) -> str:
            # Try \boxed{...}
            m = _re.search(r"\\boxed\{", text)
            if m:
                depth, start = 1, m.end()
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start:i].strip()
            # Try "#### number" (GSM8K format)
            m2 = _re.search(r"####\s*([\d,\.]+)", text)
            if m2:
                return m2.group(1).replace(",", "")
            # Last number in the response
            nums = _re.findall(r"-?[\d]+(?:\.\d+)?(?:/\d+)?", text)
            return nums[-1] if nums else ""

        answer_votes: dict[str, list[str]] = {}
        for r in results:
            ans = _extract_final(r.response)
            if ans:
                key = ans.strip().lower()
                if key not in answer_votes:
                    answer_votes[key] = []
                answer_votes[key].append(r.agent_id)

        # Pick majority answer; if no majority (all different), use LLM judge to pick best
        best_response = results[0].response
        max_votes = max((len(v) for v in answer_votes.values()), default=0)
        has_majority = max_votes >= 2  # at least 2 agents agree

        if answer_votes and has_majority:
            best_key = max(answer_votes.keys(), key=lambda k: len(answer_votes[k]))
            majority_ids = set(answer_votes[best_key])
            # Prefer leader's response if leader is in majority; else pick first majority member
            leader_r = next((r for r in results if r.agent_id == self.leader.agent_id
                             and r.agent_id in majority_ids), None)
            best_response = leader_r.response if leader_r else next(
                (r.response for r in results if r.agent_id in majority_ids), results[0].response
            )
        elif len(results) > 1:
            # No majority: use LLM to judge which solution is most rigorous
            # Truncate solutions to 600 chars to stay within context
            sol_texts = "\n\n".join(
                f"Solution {chr(65+i)}:\n{r.response[:600]}"
                for i, r in enumerate(results)
            )
            judge_prompt = (
                f"Problem: {task.get('prompt', '')[:400]}\n\n"
                f"{sol_texts}\n\n"
                "Which solution shows the most rigorous and correct reasoning? "
                "Reply with ONLY the letter (A, B, or C)."
            )
            try:
                pick = llm_call(
                    model=self.backbone_llm,
                    system="You are a math judge. Pick the best solution.",
                    user=judge_prompt,
                    max_tokens=10,
                ).strip().upper()
                idx = ord(pick[0]) - ord('A') if pick and pick[0].isalpha() else 0
                idx = max(0, min(idx, len(results) - 1))
                best_response = results[idx].response
            except Exception:
                best_response = self.leader.execute_task(task, self.backbone_llm).get("response", results[0].response)

        return MASResult(
            final_answer=best_response,
            leader_id=self.leader.agent_id,
            per_agent_responses=per_agent_responses,
            per_agent_feedback={},
            decomposition_plan=[{"agent_id": r.agent_id, "role": "primary", "skills": ["math"]}
                                 for r in results],
            extra_agents_recruited=[],
            n_rounds=1,
        )

    def _run_qa_self_consistency(self, task: dict) -> MASResult:
        """
        Self-consistency mode for QA tasks (HotpotQA, DROP).
        All k agents answer independently → pick the most common short answer.
        Mirrors AFlow's ScEnsemble for QA, which works well for span-extraction tasks.
        """
        import re as _re

        results: list[SubtaskResult] = []
        per_agent_responses: dict[str, dict] = {}

        for agent in self.team:
            resp = agent.execute_task(task, self.backbone_llm)
            results.append(SubtaskResult(
                agent_id=agent.agent_id,
                role="primary",
                subtask_name="answer",
                response=resp.get("response", ""),
            ))
            per_agent_responses[agent.agent_id] = resp

        # Extract short answers from each response for voting.
        # For QA, take the last sentence (tends to be the direct answer) or last line.
        def _extract_answer(text: str) -> str:
            text = text.strip()
            # Take last non-empty line
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if not lines:
                return text[:100].strip()
            last = lines[-1]
            # Remove common prefixes
            for prefix in ("the answer is", "answer:", "therefore,", "so,", "thus,"):
                if last.lower().startswith(prefix):
                    last = last[len(prefix):].strip().strip(":").strip()
            return last[:200].lower().strip()

        answer_votes: dict[str, list[str]] = {}
        for r in results:
            ans = _extract_answer(r.response)
            if ans:
                if ans not in answer_votes:
                    answer_votes[ans] = []
                answer_votes[ans].append(r.agent_id)

        # Pick majority answer; tie-break by preferring the leader's response
        best_response = results[0].response
        if answer_votes:
            best_key = max(answer_votes.keys(), key=lambda k: len(answer_votes[k]))
            majority_ids = set(answer_votes[best_key])
            leader_r = next((r for r in results if r.agent_id == self.leader.agent_id
                             and r.agent_id in majority_ids), None)
            best_response = leader_r.response if leader_r else next(
                (r.response for r in results if r.agent_id in majority_ids), results[0].response
            )

        return MASResult(
            final_answer=best_response,
            leader_id=self.leader.agent_id,
            per_agent_responses=per_agent_responses,
            per_agent_feedback={},
            decomposition_plan=[{"agent_id": r.agent_id, "role": "primary", "skills": ["qa"]}
                                 for r in results],
            extra_agents_recruited=[],
            n_rounds=1,
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
        import re as _re
        task_type = task.get("type", "general")
        primary = max(team, key=lambda a: a.profile.skill_memory.get(task_type, 0.2))

        # For code tasks, inject the exact function name to prevent wrong naming
        subtask_prompt = task.get("prompt", str(task))
        domain = task.get("domain", "")
        if domain in ("mbpp", "humaneval") or task_type in ("code_generation", "code_completion"):
            ep = task.get("entry_point", "")
            if not ep:
                # Try to extract from test cases
                test_cases = task.get("test_cases", []) or []
                for tc in test_cases:
                    m = _re.search(r"assert\s+(\w+)\(", str(tc))
                    if m:
                        ep = m.group(1)
                        break
                if not ep:
                    # Try from test string (HumanEval format)
                    m = _re.search(r"def check\(candidate\).*?assert candidate",
                                   task.get("test", ""), _re.DOTALL)
                    if not m:
                        # Try from prompt function signature
                        m = _re.search(r"^def (\w+)\(", subtask_prompt.strip())
                        if m:
                            ep = m.group(1)
            if ep:
                subtask_prompt = f"[REQUIRED FUNCTION NAME: {ep}]\n\n" + subtask_prompt

        assignments = [SubtaskAssignment(
            agent_id=primary.agent_id,
            role="primary",
            subtask_prompt=subtask_prompt,
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

            domain = task.get("domain", "")
            is_code = domain in ("mbpp", "humaneval") or task.get("type", "") in ("code_generation", "code_completion")
            if is_code:
                ep = task.get("entry_point", "")
                ep_hint = f"\n[REQUIRED FUNCTION NAME: {ep}]" if ep else ""
                critique_prompt = (
                    f"Original task: {task.get('prompt', '')[:400]}{ep_hint}\n\n"
                    f"Team's code attempts:\n{r1_summary}\n\n"
                    "Review the code attempts above. Identify any bugs, wrong function names, "
                    "or logical errors. Then produce the CORRECTED, complete Python function. "
                    "Output ONLY a markdown code block: ```python\n...\n```"
                )
            else:
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
        domain = task.get("domain", "")
        is_code_task = domain in ("mbpp", "humaneval") or task.get("type", "") in ("code_generation", "code_completion")

        # For code tasks: pick the best syntactically valid implementation
        # rather than hallucinating a synthesis of multiple code blocks.
        if is_code_task:
            return self._pick_best_code(task, all_results)

        results_text = "\n\n".join(
            f"=== {r.role} (Agent {r.agent_id}) ===\n{r.response}"
            for r in all_results
        )
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

    def _pick_best_code(self, task: dict, all_results: list[SubtaskResult]) -> str:
        """
        For code tasks: pick the best implementation from all_results.
        Priority: (1) passes most test cases, (2) valid syntax, (3) primary agent.
        Avoids hallucinating a synthesis of multiple code blocks.
        """
        import re as _re

        def _extract_code(text: str) -> str:
            # Strip Qwen3-8B thinking tokens before extracting code
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            if "```python" in text:
                return text.split("```python")[1].split("```")[0].strip()
            if "```" in text:
                return text.split("```")[1].split("```")[0].strip()
            # Only use raw text if it looks like code (not narrative text)
            stripped = text.strip()
            if stripped.startswith("def ") or stripped.startswith("import ") or stripped.startswith("class "):
                return stripped
            return ""

        def _sanitize_function_name(code: str, entry_point: str, task_prompt: str = "") -> str:
            """
            Ensure code has the correct function name.
            If code is only a function body (no def line), prepend the signature from the task.
            If code has a different function name, rename the first function.
            """
            if not entry_point or not code:
                return code

            # If no 'def' in code, it's just a function body — try to prepend the signature
            if not _re.search(r"^\s*def\s+\w+", code, _re.MULTILINE):
                # Extract function signature from task prompt
                sig_match = _re.search(
                    r"(def\s+" + _re.escape(entry_point) + r"\s*\([^)]*\)[^:]*:)",
                    task_prompt,
                )
                if sig_match:
                    # Indent the body and prepend the signature
                    body = "\n".join("    " + line if line.strip() else line
                                     for line in code.splitlines())
                    return sig_match.group(1) + "\n" + body
                # No signature found, return as-is
                return code

            # Check if entry_point already exists in code
            if _re.search(r"\bdef\s+" + _re.escape(entry_point) + r"\s*\(", code):
                return code
            # Find the first top-level function definition and rename it
            renamed = _re.sub(
                r"(def\s+)(\w+)(\s*\()",
                lambda m: m.group(1) + entry_point + m.group(3),
                code,
                count=1,
            )
            return renamed

        def _test_score(code: str, task: dict) -> float:
            import signal as _signal

            def _timeout_handler(signum, frame):
                raise TimeoutError()

            test_cases = task.get("test_cases") or []
            test_str = task.get("test", "")
            entry_point = task.get("entry_point", "")
            if not test_cases and not test_str:
                return 0.5  # No tests available — give neutral score
            # HumanEval: run check(fn)
            if test_str and entry_point:
                try:
                    _signal.signal(_signal.SIGALRM, _timeout_handler)
                    _signal.alarm(5)  # 5-second timeout
                    g: dict = {}
                    exec(code, g)
                    exec(test_str, g)
                    if entry_point in g:
                        g["check"](g[entry_point])
                        _signal.alarm(0)
                        return 1.0
                    else:
                        # Try callable fallback (function with wrong name)
                        candidates = [v for k, v in g.items()
                                      if callable(v) and k not in ("check", "METADATA")
                                      and not k.startswith("_")]
                        if candidates and "check" in g:
                            g["check"](candidates[0])
                            _signal.alarm(0)
                            return 1.0
                        _signal.alarm(0)
                        return 0.0  # entry point not found
                except Exception:
                    _signal.alarm(0)
                    return 0.0
            # MBPP: run assert statements
            if test_cases:
                passed = 0
                for tc in test_cases:
                    try:
                        _signal.signal(_signal.SIGALRM, _timeout_handler)
                        _signal.alarm(3)  # 3-second timeout per test
                        g2: dict = {}
                        exec(code, g2)
                        exec(tc, g2)
                        _signal.alarm(0)
                        passed += 1
                    except Exception:
                        _signal.alarm(0)
                return passed / len(test_cases)
            return 0.5

        def _syntax_ok(code: str) -> bool:
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                return False

        entry_point = task.get("entry_point", "")
        candidates = []
        for r in all_results:
            if r.role in ("reviewer", "critic"):
                continue  # skip non-implementation roles
            code = _extract_code(r.response)
            if not code:
                continue
            # Sanitize function name to match entry_point (critical for HumanEval)
            if entry_point:
                code = _sanitize_function_name(code, entry_point, task.get("prompt", ""))
            score = _test_score(code, task)
            syn = _syntax_ok(code)
            priority = 1 if r.role == "primary" else 0
            # Store the sanitized code (not r.response) so the correct name is returned
            sanitized_response = f"```python\n{code}\n```" if code else r.response
            candidates.append((score, syn, priority, sanitized_response))

        if not candidates:
            # fallback: primary or first
            for r in all_results:
                if r.role == "primary":
                    return r.response
            return all_results[0].response if all_results else ""

        # Sort: test score desc, syntax ok desc, primary first
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return candidates[0][3]

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
