"""
Self-Consistency Baseline (Wang et al., 2022).
Samples k independent responses from the same LLM, then selects the
majority-vote answer. No memory, no multi-agent collaboration.

Reference: https://arxiv.org/abs/2203.11171
"""

from __future__ import annotations

import re
import time
from collections import Counter

from ..llm import llm_call


class SelfConsistencyPool:
    """
    Self-Consistency: sample k responses, pick majority answer.
    Uses the same format hints as EvoPool agents for fair comparison.
    """

    def __init__(
        self,
        k: int = 5,
        backbone_llm: str = "qwen3-8b",
        temperature: float = 0.8,
        seed: int = 42,
    ):
        self.k = k
        self.backbone_llm = backbone_llm
        self.temperature = temperature
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
                hint += "\n\nTest cases (sample):\n" + "\n".join(str(tc)[:500] for tc in test_cases[:3])
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
                + "\n\nSolve step by step. Express your final answer in LaTeX inside \\boxed{}."
                + " For example: \\boxed{\\frac{3}{5}} or \\boxed{42} or \\boxed{x+1}."
            )
        elif domain in ("hotpotqa", "drop") or task_type in ("multi_hop_qa", "reading_comprehension"):
            user = base_prompt + "\n\nProvide a concise, direct answer."
        else:
            user = base_prompt

        return system, user

    def _extract_answer(self, response: str, task: dict) -> str:
        """Extract the key answer string for majority voting."""
        domain = task.get("domain", "")
        task_type = task.get("type", "")

        if domain == "gsm8k" or task_type == "math_word_problem":
            m = re.search(r"####\s*([\d,\.]+)", response)
            return m.group(1).replace(",", "").rstrip(".") if m else response[-100:]

        if domain == "math" or task_type in ("math_competition", "arithmetic"):
            # Extract \boxed{...}
            m = re.search(r"\\boxed\{", response)
            if m:
                depth, start = 1, m.end()
                for i in range(start, len(response)):
                    if response[i] == "{":
                        depth += 1
                    elif response[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return response[start:i].strip()
            nums = re.findall(r"-?[\d]+(?:\.\d+)?", response)
            return nums[-1] if nums else response[-50:]

        if domain in ("mbpp", "humaneval") or task_type in ("code_generation", "code_completion"):
            # For code, return the full response (no majority voting — use first pass)
            return response

        # QA tasks: return last sentence or short answer
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[-1] if lines else response[-100:]

    def process_task(self, task: dict, evaluator) -> dict:
        """Sample k responses, majority vote, evaluate."""
        system, user = self._build_prompt(task)

        responses_raw = []
        for _ in range(self.k):
            try:
                resp = llm_call(
                    model=self.backbone_llm,
                    system=system,
                    user=user,
                    max_tokens=1024,
                    temperature=self.temperature,
                )
                responses_raw.append(resp)
            except Exception:
                pass

        if not responses_raw:
            responses_raw = [""]

        domain = task.get("domain", "")
        task_type = task.get("type", "")
        is_code = domain in ("mbpp", "humaneval") or task_type in ("code_generation", "code_completion")

        if is_code:
            # For code: evaluate all, pick best
            best_response = responses_raw[0]
            best_score = -1.0
            for r in responses_raw:
                s = evaluator(task, {"sc": {"agent_id": "sc", "response": r, "task_type": task_type}})
                sc = s.get("team_score", 0.0)
                if sc > best_score:
                    best_score = sc
                    best_response = r
            final_response = best_response
        else:
            # Majority vote on extracted answers
            extracted = [self._extract_answer(r, task) for r in responses_raw]
            vote_counts = Counter(extracted)
            majority_answer = vote_counts.most_common(1)[0][0]
            # Find the full response corresponding to majority answer
            for r, e in zip(responses_raw, extracted):
                if e == majority_answer:
                    final_response = r
                    break
            else:
                final_response = responses_raw[0]

        responses = {
            "self_consistency": {
                "agent_id": "self_consistency",
                "response": final_response,
                "task_type": task.get("type", "unknown"),
            }
        }
        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.0)

        self.task_index += 1
        metrics = {
            "task_index": self.task_index,
            "team_score": team_score,
            "pool_size": 1,
            "profile_diversity": 0.0,
            "timestamp": time.time(),
        }
        self.metrics_log.append(metrics)

        return {
            "task_id": task.get("id", self.task_index),
            "team_score": team_score,
            "team_agent_ids": ["self_consistency"],
            "metrics": metrics,
        }
