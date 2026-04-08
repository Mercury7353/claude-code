"""
DyLAN Baseline: Dynamic LLM-Agent Network (COLM 2024).
Official codebase: https://github.com/SALT-NLP/DyLAN

Adapted to EvoPool's streaming task interface.
Key changes from the official code:
  - LLM calls redirected to local vLLM server (qwen3-8b) via patched utils.py
  - Task format adapter: our dict → DyLAN question string
  - process_task() interface to match EvoPool's benchmark runner
  - qtype="math_exp" for GSM8K/MATH; "open-ended" for other domains

Original DyLAN key features preserved:
  - Multi-round multi-agent debate (LLMLP.forward)
  - Agent Importance Score (AIS) via listwise ranking
  - Early stopping when 2/3 consensus is reached
  - No cross-task memory (static pool)
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path


# Add official DyLAN codebase to path
_DYLAN_CODE_PATH = str(Path(__file__).parent.parent.parent / "external_baselines" / "DyLAN" / "code" / "demo")
if _DYLAN_CODE_PATH not in sys.path:
    sys.path.insert(0, _DYLAN_CODE_PATH)


class DyLANPool:
    """
    Wrapper around official DyLAN LLMLP to match EvoPool's process_task interface.
    """

    def __init__(
        self,
        pool_size: int = 10,
        team_size: int = 3,
        backbone_llm: str = "qwen3-8b",
        seed: int = 42,
        rounds: int = 3,
    ):
        self.pool_size = pool_size
        self.team_size = team_size
        self.backbone_llm = backbone_llm
        self.rounds = rounds
        self.task_index = 0
        self.metrics_log: list[dict] = []
        self._system = None  # Lazy-init (imports patched utils)

    def _get_system(self, qtype: str = "math_exp"):
        """Lazy-initialize LLMLP to ensure the sys.path patch is active."""
        from LLMLP import LLMLP
        from prompt_lib import ROLE_MAP, ROLE_MAP_MATH

        role_map = ROLE_MAP_MATH if qtype == "math_exp" else ROLE_MAP
        # Use 'Math Reasoning' roles for numeric tasks; fall back to first N roles
        available_roles = list(role_map.keys())[:self.team_size]
        if len(available_roles) < self.team_size:
            available_roles += [available_roles[-1]] * (self.team_size - len(available_roles))

        return LLMLP(
            default_model_name=self.backbone_llm,
            agents=self.team_size,
            agent_roles=available_roles,
            rounds=self.rounds,
            activation="listwise",
            qtype=qtype,
            mtype=self.backbone_llm,
        )

    def _infer_qtype(self, task: dict) -> str:
        domain = task.get("domain", "")
        task_type = task.get("type", "")
        if (domain in ("gsm8k", "math", "math_hard") or task_type in ("arithmetic", "math", "math_competition_hard", "aime_problem")
                or domain.startswith("aime_")):
            return "math_exp"
        return "open-ended"

    def _task_to_question(self, task: dict) -> str:
        """Convert our task dict to a question string for DyLAN."""
        prompt = task.get("prompt", str(task.get("question", str(task))))
        # For code tasks, include function name and test cases
        domain = task.get("domain", "")
        if domain in ("mbpp", "humaneval") or task.get("type") in ("code_generation", "code_completion"):
            import re as _re
            entry_point = task.get("entry_point", "")
            if not entry_point:
                for tc in (task.get("test_cases") or []):
                    _m = _re.search(r"assert\s+(\w+)\s*\(", str(tc))
                    if _m:
                        entry_point = _m.group(1)
                        break
            if entry_point:
                prompt = f"[REQUIRED FUNCTION NAME: {entry_point}]\n\n" + prompt
            test_cases = task.get("test_cases", [])
            if test_cases:
                prompt += "\n\nTest cases:\n" + "\n".join(str(tc) for tc in test_cases[:3])
        return prompt

    def process_task(self, task: dict, evaluator) -> dict:
        """
        DyLAN task processing using the official LLMLP forward pass:
        1. Infer question type from task domain
        2. Run LLMLP.forward(question) — multi-round debate with AIS
        3. Get consensus answer
        4. Evaluate and log
        """
        qtype = self._infer_qtype(task)
        system = self._get_system(qtype)

        question = self._task_to_question(task)
        # Official DyLAN forward: returns (answer, resp_cnt, completions, prompt_tokens, completion_tokens)
        result = system.forward(question)
        answer = result[0] if isinstance(result, tuple) else result

        # Wrap in response format expected by evaluator
        responses = {
            "dylan_ensemble": {
                "agent_id": "dylan_ensemble",
                "response": str(answer) if answer is not None else "",
                "task_type": task.get("type", "unknown"),
            }
        }

        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.5)

        self.task_index += 1
        metrics = {
            "task_index": self.task_index,
            "team_score": team_score,
            "pool_size": self.pool_size,
            "profile_diversity": 0.0,
        }
        self.metrics_log.append(metrics)

        return {
            "task_id": task.get("id", self.task_index),
            "team_score": team_score,
            "team_agent_ids": [f"dylan_agent_{i}" for i in range(self.team_size)],
            "metrics": metrics,
        }
