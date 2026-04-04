"""
AFlow Baseline: Automated Agentic Workflow Generation.
Paper: "AFlow: Automating Agentic Workflow Generation" (NeurIPS 2024)
arXiv: 2410.10762  |  Official repo: https://github.com/FoundationAgents/AFlow

Uses the OFFICIAL AFlow operators (Custom, ScEnsemble, Review, Revise,
CustomCodeGenerate, AnswerGenerate) from the open-source codebase.

Since MCTS search is too expensive at inference time, we fix domain-appropriate
workflows based on AFlow paper Table 2 findings:
  math  (GSM8K, MATH)        : ScEnsemble(n=3) → AnswerGenerate (best for math)
  code  (MBPP, HumanEval)    : CustomCodeGenerate → Review → Revise
  qa    (HotpotQA, DROP)     : Custom → ScEnsemble(n=2)

All LLM calls are redirected to local vLLM server via patched async_llm.py.
No cross-task learning — this is a strong STATIC baseline.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Callable

# Add official AFlow codebase to path
_AFLOW_ROOT = str(Path(__file__).parent.parent.parent / "external_baselines" / "AFlow")
if _AFLOW_ROOT not in sys.path:
    sys.path.insert(0, _AFLOW_ROOT)

# Lazy imports — deferred until first use so missing deps fail gracefully
_ops = None


def _get_ops():
    global _ops
    if _ops is not None:
        return _ops
    from scripts.async_llm import AsyncLLM, LLMConfig
    from scripts.operators import Custom, ScEnsemble, Review, Revise, AnswerGenerate, CustomCodeGenerate

    # vLLM config (async_llm.py already patches base_url, but pass explicit config)
    cfg = LLMConfig({
        "model": "qwen3-8b",
        "temperature": 0.7,
        "key": "no-key",
        "base_url": "http://10.217.117.45:8000/v1",
        "top_p": 1,
    })
    llm = AsyncLLM(cfg)

    _ops = {
        "llm": llm,
        "custom": Custom(llm),
        "sc_ensemble": ScEnsemble(llm),
        "review": Review(llm),
        "revise": Revise(llm),
        "answer_generate": AnswerGenerate(llm),
        "code_generate": CustomCodeGenerate(llm),
    }
    return _ops


# ---------------------------------------------------------------------------
# Domain routing
# ---------------------------------------------------------------------------

_DOMAIN_TO_CATEGORY = {
    "gsm8k": "math",
    "math": "math",
    "mbpp": "code",
    "humaneval": "code",
    "hotpotqa": "qa",
    "drop": "qa",
}


async def _workflow_math(ops: dict, prompt: str) -> str:
    """AFlow math workflow: ScEnsemble(3) → AnswerGenerate."""
    # Generate 3 independent chain-of-thought solutions
    solutions = []
    for _ in range(3):
        resp = await ops["custom"](input=prompt, instruction="Solve this math problem step by step:\n")
        solutions.append(resp.get("response", ""))

    # Self-consistency ensemble
    best = await ops["sc_ensemble"](solutions=solutions, problem=prompt)
    return best.get("response", solutions[0])


async def _workflow_code(ops: dict, prompt: str, entry_point: str = "solution") -> str:
    """AFlow code workflow: CustomCodeGenerate → Review → Revise."""
    # Generate initial code solution
    draft = await ops["code_generate"](problem=prompt, entry_point=entry_point, instruction="")
    code = draft.get("response", "")

    # Review the code
    review = await ops["review"](problem=prompt, solution=code)
    feedback = review.get("feedback", "")
    approved = review.get("review_result", False)

    if not approved and feedback:
        # Revise based on feedback
        revised = await ops["revise"](problem=prompt, solution=code, feedback=feedback)
        code = revised.get("solution", code)

    return code


async def _workflow_qa(ops: dict, prompt: str) -> str:
    """AFlow QA workflow: Custom → ScEnsemble(2)."""
    # Two independent reasoning attempts
    resp1 = await ops["custom"](input=prompt, instruction="Answer this question carefully:\n")
    resp2 = await ops["custom"](
        input=prompt,
        instruction="Think step by step and answer this question:\n",
    )
    solutions = [resp1.get("response", ""), resp2.get("response", "")]

    best = await ops["sc_ensemble"](solutions=solutions, problem=prompt)
    return best.get("response", solutions[0])


def _infer_entry_point(task: dict) -> str:
    """Infer function name from task for MBPP/HumanEval."""
    if task.get("entry_point"):
        return task["entry_point"]
    # Try to extract from test cases (e.g. "assert func_name(...)")
    import re
    for test in task.get("test_cases", []):
        m = re.match(r"assert\s+(\w+)\s*\(", test)
        if m:
            return m.group(1)
    # Try to extract from prompt
    prompt = task.get("prompt", "")
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    if m:
        return m.group(1)
    return "solution"


async def _run_workflow_async(category: str, ops: dict, task: dict) -> str:
    prompt = task.get("prompt", str(task))
    entry_point = _infer_entry_point(task)

    if category == "math":
        return await _workflow_math(ops, prompt)
    elif category == "code":
        return await _workflow_code(ops, prompt, entry_point)
    else:
        return await _workflow_qa(ops, prompt)


# ---------------------------------------------------------------------------
# AFlowPool: interface compatible with EvoPool benchmark runner
# ---------------------------------------------------------------------------

class AFlowPool:
    """
    AFlow static-workflow baseline using the official AFlow operators.

    Uses domain-optimal workflows discovered by AFlow's MCTS (per paper Table 2).
    No pool, no memory, no cross-task learning — purely static per-task workflow.
    """

    def __init__(
        self,
        pool_size: int = 20,   # unused
        team_size: int = 3,    # unused
        backbone_llm: str = "qwen3-8b",
        seed: int = 42,
    ):
        self.backbone_llm = backbone_llm
        self.seed = seed
        self.task_index = 0
        self.metrics_log: list[dict] = []
        self._ops = None  # lazy init

    def _get_ops(self):
        if self._ops is None:
            self._ops = _get_ops()
        return self._ops

    def process_task(self, task: dict, evaluator: Callable) -> dict:
        domain = task.get("domain", task.get("type", "qa")).lower()
        category = _DOMAIN_TO_CATEGORY.get(domain, "qa")
        ops = self._get_ops()

        # Bridge async → sync
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        answer = loop.run_until_complete(_run_workflow_async(category, ops, task))

        responses = {
            "aflow_agent": {
                "agent_id": "aflow_agent",
                "response": answer,
                "task_type": domain,
            }
        }

        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.0)

        self.task_index += 1
        metrics = {
            "task_index": self.task_index,
            "team_score": team_score,
            "domain": domain,
            "category": category,
            "workflow": f"aflow_{category}",
            "pool_size": 1,
            "profile_diversity": 0.0,
        }
        self.metrics_log.append(metrics)
        return {
            "task_id": task.get("id", self.task_index),
            "team_score": team_score,
            "team_agent_ids": ["aflow_agent"],
            "metrics": metrics,
        }
