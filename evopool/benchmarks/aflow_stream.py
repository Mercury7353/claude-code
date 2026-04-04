"""
AFlow sequential benchmark stream.
Streams tasks from multiple AFlow benchmarks in sequence:
  GSM8K → HotpotQA → MBPP → MATH → HumanEval → DROP

Each task dict has:
  {"id": str, "type": str, "prompt": str, "answer": str/list, "domain": str}
"""

from __future__ import annotations

import random
from typing import Iterator

from datasets import load_dataset


AFLOW_DOMAINS = ["gsm8k", "hotpotqa", "mbpp", "math", "humaneval", "drop"]

DOMAIN_TASK_TYPE_MAP = {
    "gsm8k": "math_word_problem",
    "hotpotqa": "multi_hop_qa",
    "mbpp": "code_generation",
    "math": "math_competition",
    "humaneval": "code_completion",
    "drop": "reading_comprehension",
}


def load_aflow_stream(
    n_per_domain: int = 10,
    domains: list[str] | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Load a sequential stream of tasks from AFlow benchmarks.

    Args:
        n_per_domain: Number of tasks to sample per domain
        domains: List of domains to include (default: all 6)
        shuffle: Whether to shuffle within each domain
        seed: Random seed

    Returns:
        List of task dicts ordered by domain
    """
    if domains is None:
        domains = AFLOW_DOMAINS

    rng = random.Random(seed)
    all_tasks = []

    for domain in domains:
        tasks = _load_domain(domain, n_per_domain, rng, shuffle)
        all_tasks.extend(tasks)

    return all_tasks


def _load_domain(domain: str, n: int, rng: random.Random, shuffle: bool) -> list[dict]:
    """Load n tasks from a single domain."""
    tasks = []

    try:
        if domain == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            samples = list(ds)
            if shuffle:
                rng.shuffle(samples)
            for i, item in enumerate(samples[:n]):
                tasks.append({
                    "id": f"gsm8k_{i}",
                    "type": DOMAIN_TASK_TYPE_MAP[domain],
                    "domain": domain,
                    "prompt": f"Solve this math problem step by step:\n{item['question']}",
                    "answer": item["answer"],
                })

        elif domain == "hotpotqa":
            ds = load_dataset("hotpot_qa", "distractor", split="validation")
            samples = list(ds)
            if shuffle:
                rng.shuffle(samples)
            for i, item in enumerate(samples[:n]):
                context = "\n".join([f"[{t}]: {' '.join(s)}" for t, s in zip(item["context"]["title"], item["context"]["sentences"])])
                tasks.append({
                    "id": f"hotpotqa_{i}",
                    "type": DOMAIN_TASK_TYPE_MAP[domain],
                    "domain": domain,
                    "prompt": f"Answer the question based on the following context:\n{context}\n\nQuestion: {item['question']}",
                    "answer": item["answer"],
                })

        elif domain == "mbpp":
            ds = load_dataset("mbpp", split="test")
            samples = list(ds)
            if shuffle:
                rng.shuffle(samples)
            for i, item in enumerate(samples[:n]):
                tasks.append({
                    "id": f"mbpp_{i}",
                    "type": DOMAIN_TASK_TYPE_MAP[domain],
                    "domain": domain,
                    "prompt": f"Write a Python function:\n{item['text']}\n\nTest cases:\n" + "\n".join(item["test_list"]),
                    "answer": item["code"],
                    "test_cases": item["test_list"],
                })

        elif domain == "humaneval":
            ds = load_dataset("openai_humaneval", split="test")
            samples = list(ds)
            if shuffle:
                rng.shuffle(samples)
            for i, item in enumerate(samples[:n]):
                tasks.append({
                    "id": f"humaneval_{i}",
                    "type": DOMAIN_TASK_TYPE_MAP[domain],
                    "domain": domain,
                    "prompt": item["prompt"],
                    "answer": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                })

        elif domain == "math":
            # Try primary dataset, fall back to EleutherAI mirror (multi-subject)
            try:
                ds = load_dataset("hendrycks/competition_math", split="test")
                samples = list(ds)
            except Exception:
                _subjects = ["algebra", "counting_and_probability", "geometry",
                             "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
                samples = []
                for _subj in _subjects:
                    try:
                        _ds = load_dataset("EleutherAI/hendrycks_math", _subj, split="test")
                        samples.extend(list(_ds))
                    except Exception:
                        pass
                if not samples:
                    raise RuntimeError("Could not load any math dataset")
            if shuffle:
                rng.shuffle(samples)
            for i, item in enumerate(samples[:n]):
                tasks.append({
                    "id": f"math_{i}",
                    "type": DOMAIN_TASK_TYPE_MAP[domain],
                    "domain": domain,
                    "prompt": f"Solve this competition math problem:\n{item['problem']}",
                    "answer": item["solution"],
                })

        elif domain == "drop":
            ds = load_dataset("drop", split="validation")
            samples = list(ds)
            if shuffle:
                rng.shuffle(samples)
            for i, item in enumerate(samples[:n]):
                tasks.append({
                    "id": f"drop_{i}",
                    "type": DOMAIN_TASK_TYPE_MAP[domain],
                    "domain": domain,
                    "prompt": f"Passage: {item['passage']}\nQuestion: {item['question']}",
                    "answer": item["answers_spans"]["spans"][0] if item["answers_spans"]["spans"] else "",
                })

    except Exception as e:
        # Fallback to dummy tasks if dataset not available
        print(f"Warning: Could not load {domain} dataset: {e}. Using dummy tasks.")
        for i in range(n):
            tasks.append({
                "id": f"{domain}_{i}",
                "type": DOMAIN_TASK_TYPE_MAP[domain],
                "domain": domain,
                "prompt": f"[{domain}] Task {i}: placeholder task for testing",
                "answer": "placeholder_answer",
            })

    return tasks


class AFlowEvaluator:
    """Evaluate agent responses on AFlow tasks."""

    def __call__(self, task: dict, responses: dict[str, dict]) -> dict:
        """
        Evaluate all agent responses and return scores.
        Returns {agent_id: score, "team_score": float}
        """
        scores = {}
        for agent_id, response_dict in responses.items():
            response_text = response_dict.get("response", "")
            score = self._evaluate_response(task, response_text)
            scores[agent_id] = score

        # Team score = max of individual scores (best answer wins)
        team_score = max(scores.values()) if scores else 0.0
        scores["team_score"] = team_score
        return scores

    def _evaluate_response(self, task: dict, response: str) -> float:
        """Evaluation. Returns score 0–1."""
        import re as _re
        answer = str(task.get("answer", "")).strip()
        response_lower = response.lower().strip()
        domain = task.get("domain", "")

        if not answer or answer.lower() == "placeholder_answer":
            return 0.5  # Neutral for dummy tasks

        # For code tasks, try to extract and test
        if domain in ("mbpp", "humaneval"):
            return self._evaluate_code(task, response)

        # GSM8K: reference answer is full solution ending with "#### <number>"
        # Extract just the final number for evaluation
        if domain == "gsm8k":
            match = _re.search(r"####\s*([\d,\.]+)", answer)
            if match:
                final_num = match.group(1).replace(",", "")
                # Check if response contains this number
                response_nums = _re.findall(r"[\d,\.]+", response)
                response_nums_clean = {n.replace(",", "").rstrip(".") for n in response_nums}
                return 1.0 if final_num.rstrip(".") in response_nums_clean else 0.0

        # MATH: reference is full solution. Extract boxed answer or last number.
        if domain == "math":
            # Extract \boxed{...} — handle nested braces by counting depth
            def _extract_boxed(s: str) -> str:
                m = _re.search(r"\\boxed\{", s)
                if not m:
                    return ""
                depth, start = 1, m.end()
                for i in range(start, len(s)):
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                        if depth == 0:
                            return s[start:i].strip()
                return s[start:].strip()

            target = _extract_boxed(answer)
            if not target:
                # Fallback: last number in answer
                nums = _re.findall(r"-?[\d]+(?:\.\d+)?(?:/\d+)?", answer)
                target = nums[-1] if nums else ""
            if target:
                # Normalize: remove LaTeX formatting
                target_clean = _re.sub(r"\\[a-zA-Z]+\{?|[{}\\]", "", target).strip().lower()
                response_clean = response_lower
                return 1.0 if (target_clean in response_clean or target in response_lower) else 0.0

        # DROP: answers are short numeric or text spans
        if domain == "drop":
            answer_lower = answer.lower().strip()
            if answer_lower in response_lower:
                return 1.0
            return 0.0

        # HotpotQA: short text answer
        if domain == "hotpotqa":
            answer_lower = answer.lower().strip()
            if answer_lower in response_lower:
                return 1.0
            # Partial word overlap for short answers
            answer_words = set(answer_lower.split())
            response_words = set(response_lower.split())
            if answer_words:
                overlap = len(answer_words & response_words) / len(answer_words)
                return min(overlap, 1.0)
            return 0.0

        # Generic fallback: exact match then word overlap
        answer_lower = answer.lower().strip()
        if answer_lower in response_lower:
            return 1.0
        answer_words = set(answer_lower.split())
        response_words = set(response_lower.split())
        overlap = len(answer_words & response_words) / max(len(answer_words), 1)
        return min(overlap, 1.0)

    def _evaluate_code(self, task: dict, response: str) -> float:
        """Try to execute code and check test cases."""
        domain = task.get("domain", "")

        # Extract code block from response
        code = response
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]

        if domain == "humaneval":
            # HumanEval: run check(entry_point_fn) to actually test correctness
            test_str = task.get("test", "")
            entry_point = task.get("entry_point", "")
            if not test_str:
                return 0.5
            try:
                exec_globals: dict = {}
                exec(code, exec_globals)
                exec(test_str, exec_globals)
                if entry_point and entry_point in exec_globals:
                    exec_globals["check"](exec_globals[entry_point])
                elif "check" in exec_globals:
                    # Try to find the function by inspecting exec_globals
                    candidates = [v for k, v in exec_globals.items()
                                  if callable(v) and k not in ("check",) and not k.startswith("_")]
                    if candidates:
                        exec_globals["check"](candidates[0])
                return 1.0
            except Exception:
                return 0.0

        # MBPP and other code tasks: test_cases are direct assert statements
        test_cases = task.get("test_cases", []) or [task.get("test", "")]
        if not test_cases:
            return 0.5

        passed = 0
        total = len([t for t in test_cases if t])
        for test in test_cases:
            if not test:
                continue
            try:
                exec_globals: dict = {}
                exec(code, exec_globals)
                exec(test, exec_globals)
                passed += 1
            except Exception:
                pass
        return passed / max(total, 1)
