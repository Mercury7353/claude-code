"""
Single-Agent Baseline.
A single Qwen3-8B agent answers every task with no pool, no memory, no collaboration.
This is the simplest possible baseline — serves as lower bound showing the value of
the entire EvoPool machinery (pool + team selection + MAS + CoDream + Lifecycle).

The agent uses the same domain-specific format hints as EvoPool agents for fair comparison.
"""

from __future__ import annotations

import time

from ..llm import llm_call


class SingleAgentPool:
    """
    Single agent, no memory, no collaboration.
    Same prompt formatting as EvoPool for apples-to-apples comparison.
    """

    def __init__(
        self,
        backbone_llm: str = "qwen3-8b",
        seed: int = 42,
    ):
        self.backbone_llm = backbone_llm
        self.task_index = 0
        self.metrics_log: list[dict] = []

    def _build_prompt(self, task: dict) -> tuple[str, str]:
        """Return (system, user) prompts with domain-specific format hints."""
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        base_prompt = task.get("prompt", task.get("question", str(task)))

        system = "You are a helpful AI assistant."

        if domain in ("mbpp", "humaneval") or task_type in ("code_generation", "code_completion"):
            system = "You are an expert Python programmer. Write clean, correct, and efficient code."
            # Include entry_point and test cases so model uses correct function name
            entry_point = task.get("entry_point", "")
            if not entry_point:
                import re as _re
                for tc in (task.get("test_cases") or []):
                    _m = _re.search(r"assert\s+(\w+)\s*\(", str(tc))
                    if _m:
                        entry_point = _m.group(1)
                        break
            hint = ""
            if entry_point:
                hint += f"\n\n[REQUIRED FUNCTION NAME: {entry_point}]"
            test_cases = task.get("test_cases", [])
            if test_cases:
                hint += "\n\nTest cases:\n" + "\n".join(str(tc) for tc in test_cases[:3])
            user = (
                base_prompt + hint
                + "\n\nIMPORTANT: Output ONLY the complete Python function implementation "
                "in a markdown code block (```python ... ```) with no explanation outside the block."
            )
        elif domain == "gsm8k" or task_type == "math_word_problem":
            user = base_prompt + "\n\nSolve step by step. End your answer with: #### <final number>"
        elif domain == "math" or task_type in ("math_competition", "arithmetic"):
            user = (
                base_prompt
                + "\n\nSolve step by step. Before finalizing, verify your answer by checking it "
                "against the original problem or using an alternative approach. "
                "Express your final answer in LaTeX inside \\boxed{}. "
                "For example: \\boxed{\\frac{3}{5}} or \\boxed{42} or \\boxed{x+1}."
            )
        else:
            user = base_prompt

        return system, user

    def process_task(self, task: dict, evaluator) -> dict:
        t0 = time.time()
        system, user = self._build_prompt(task)

        # Hard math tasks (AIME / competition math) need extended thinking for long reasoning chains.
        is_hard_math = task.get("type") in ("aime_problem", "math_competition_hard") or task.get("domain", "").startswith("aime_") or task.get("domain") == "math_hard"
        response = llm_call(
            model=self.backbone_llm,
            system=system,
            user=user,
            max_tokens=4096 if is_hard_math else (1024 if task.get("domain") in ("gsm8k", "math") else 512),
            enable_thinking=is_hard_math,
        )

        agent_id = "single_agent"
        evaluation = evaluator(task, {agent_id: {"agent_id": agent_id, "response": response}})
        score = evaluation.get("team_score", 0.0)

        self.task_index += 1
        elapsed = time.time() - t0
        self.metrics_log.append({
            "task_index": self.task_index,
            "score": score,
            "elapsed": elapsed,
            "domain": task.get("domain", ""),
        })

        if self.task_index % 10 == 0:
            recent = [m["score"] for m in self.metrics_log[-10:]]
            total_elapsed = sum(m["elapsed"] for m in self.metrics_log)
            print(
                f"  Task {self.task_index}/600 | "
                f"Recent mean: {sum(recent)/len(recent):.3f} | "
                f"Elapsed: {total_elapsed:.0f}s"
            )

        return {
            "task_index": self.task_index,
            "team_score": score,   # matches EvoPool interface expected by run_experiment.py
            "score": score,
            "domain": task.get("domain", ""),
            "response": response,
        }

    def get_results(self) -> dict:
        scores = [m["score"] for m in self.metrics_log]
        total_elapsed = sum(m["elapsed"] for m in self.metrics_log)
        mean_score = sum(scores) / len(scores) if scores else 0.0

        # AUC = mean score (no learning curve for static agent)
        auc = mean_score

        # Per-domain scores
        domain_scores: dict[str, list[float]] = {}
        for m in self.metrics_log:
            d = m["domain"]
            domain_scores.setdefault(d, []).append(m["score"])
        per_domain = {d: sum(v) / len(v) for d, v in domain_scores.items()}

        print(f"\n=== Results ===")
        print(f"Mean score: {mean_score:.3f}")
        print(f"AUC: {auc:.3f}")
        print(f"Elapsed: {total_elapsed:.0f}s")
        for d, s in per_domain.items():
            print(f"  {d}: {s:.3f}")

        return {
            "mean_score": mean_score,
            "auc": auc,
            "per_domain": per_domain,
            "per_task_results": self.metrics_log,
            "elapsed": total_elapsed,
        }
