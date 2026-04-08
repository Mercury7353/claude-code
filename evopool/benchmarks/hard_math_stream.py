"""
Hard Math Stream benchmark.

Streams competition math problems at increasing difficulty:
  MATH-hard (L4+L5) → AIME 2022 → AIME 2023 → AIME 2024 → AIME 2025

Each task dict:
  {"id": str, "type": "math_competition_hard", "domain": str,
   "prompt": str, "answer": str, "source": str}

Domains: "math_hard", "aime_2022", "aime_2023", "aime_2024", "aime_2025"
"""

from __future__ import annotations

import random
import re
from typing import Union

from datasets import load_dataset


HARD_MATH_DOMAINS = ["math_hard", "aime_2022", "aime_2023", "aime_2024", "aime_2025", "aime_2026"]


def load_hard_math_stream(
    domains: list[str] | None = None,
    n_per_domain: int | dict | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> list[dict]:
    """
    Load a stream of hard competition math problems.

    Args:
        domains: subset of HARD_MATH_DOMAINS (default: all)
        n_per_domain: int (same for all) or dict per domain or None (use all available)
        seed: random seed
        shuffle: whether to shuffle within each domain

    Returns:
        List of task dicts in domain order
    """
    if domains is None:
        domains = HARD_MATH_DOMAINS

    if n_per_domain is None:
        n_per_domain = {}
    elif isinstance(n_per_domain, int):
        n_per_domain = {d: n_per_domain for d in domains}

    rng = random.Random(seed)
    all_tasks = []

    for domain in domains:
        n = n_per_domain.get(domain, None)  # None = use all
        tasks = _load_domain(domain, n, rng, shuffle)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} tasks from {domain}")

    return all_tasks


def _load_domain(domain: str, n: int | None, rng: random.Random, shuffle: bool) -> list[dict]:
    tasks = []

    try:
        if domain == "math_hard":
            # MATH-500 level 4 and 5 only
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            samples = [x for x in ds if x["level"] >= 4]
            if shuffle:
                rng.shuffle(samples)
            if n is not None:
                samples = samples[:n]
            for i, item in enumerate(samples):
                subtype = item.get("subject", "").lower().replace(" ", "_")
                tasks.append({
                    "id": f"math_hard_{i}",
                    "type": "math_competition_hard",
                    "domain": domain,
                    "subtype": subtype,
                    "level": item.get("level", 4),
                    "prompt": (
                        f"Solve this competition math problem. "
                        f"Put your final answer in \\boxed{{}}.\n\n{item['problem']}"
                    ),
                    "answer": item["answer"],
                    "solution": item.get("solution", ""),
                })

        elif domain.startswith("aime_"):
            year_str = domain.split("_")[1]  # "2022", "2023", "2024", "2025"
            year = int(year_str)

            if year <= 2024:
                # AI-MO/aimo-validation-aime has 2022, 2023, 2024
                ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
                samples = [
                    x for x in ds
                    if re.search(str(year), x.get("url", ""))
                ]
            elif year == 2025:
                # AIME 2025: TIGER-Lab/AIME25
                ds = load_dataset("TIGER-Lab/AIME25", split="train")
                samples = list(ds)
            else:
                # AIME 2026: try di-zhang-fdu/AIME_2026 or fallback
                try:
                    ds = load_dataset("di-zhang-fdu/AIME_2026", split="train")
                    samples = list(ds)
                except Exception:
                    # Try alternate dataset name
                    ds = load_dataset("Maxwell-Jia/AIME_2026", split="train")
                    samples = list(ds)

            if shuffle:
                rng.shuffle(samples)
            if n is not None:
                samples = samples[:n]

            for i, item in enumerate(samples):
                # Handle different column names
                problem = item.get("problem", item.get("question", ""))
                answer = str(item.get("answer", "")).strip()
                tasks.append({
                    "id": f"{domain}_{i}",
                    "type": "aime_problem",
                    "domain": domain,
                    "subtype": "aime",
                    "year": year,
                    "prompt": (
                        f"Solve this AIME problem. The answer is an integer from 0 to 999. "
                        f"Show your work step by step, then write: The answer is: <integer>\n\n"
                        f"{problem}"
                    ),
                    "answer": answer,
                })

    except Exception as e:
        print(f"Warning: Could not load {domain}: {e}. Using 0 tasks.")

    return tasks


class HardMathEvaluator:
    """Evaluate responses on hard math benchmarks."""

    def __call__(self, task: dict, responses: dict[str, dict]) -> dict:
        scores = {}
        for agent_id, response_dict in responses.items():
            response = response_dict.get("response", "")
            score = self._evaluate(task, response)
            scores[agent_id] = score
        scores["team_score"] = max(scores.values()) if scores else 0.0
        return scores

    def _evaluate(self, task: dict, response: str) -> float:
        # Strip thinking tokens
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        domain = task.get("domain", "")
        answer = str(task.get("answer", "")).strip()

        if not answer:
            return 0.0

        if domain.startswith("aime_"):
            return self._eval_aime(answer, response)
        else:
            return self._eval_math(answer, response)

    def _eval_aime(self, expected: str, response: str) -> float:
        """AIME answer is integer 0-999. Try multiple extraction strategies."""
        expected = expected.strip()

        def _check(pred: str) -> bool:
            return pred.lstrip("0") == expected.lstrip("0") or pred == expected

        # Collect all candidate answers, check each — don't early-exit on wrong match
        candidates = []

        # Strategy 1: explicit "The answer is: X" or "Final answer: X" (high confidence)
        for pat in [r"[Tt]he\s+answer\s+is:?\s*(\d+)", r"[Ff]inal\s+answer:?\s*(\d+)"]:
            matches = re.findall(pat, response)
            if matches:
                candidates.append(("explicit", matches[-1]))

        # Strategy 2: \boxed{integer}
        boxed = re.findall(r"\\boxed\{(\d{1,3})\}", response)
        if boxed:
            candidates.append(("boxed", boxed[-1]))

        # Strategy 3: bare "answer: N" in last 300 chars
        m = re.search(r"[Aa]nswer:?\s*(\d+)", response[-300:])
        if m:
            candidates.append(("bare", m.group(1)))

        # Strategy 4: last 1-3 digit integer in tail
        tail = response[-600:]
        nums = re.findall(r"\b(\d{1,3})\b", tail)
        if nums:
            candidates.append(("tail", nums[-1]))

        # Check candidates in priority order — return 1.0 if ANY matches
        for _, pred in candidates:
            if _check(pred):
                return 1.0

        # If we had candidates but none matched, return 0.0
        if candidates:
            return 0.0
        if nums:
            return 0.0
        return 0.0

    def _eval_math(self, expected: str, response: str) -> float:
        """MATH-hard: check \\boxed{} answer."""
        # Extract boxed answer
        boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", response)
        if boxed:
            pred = boxed[-1].strip()
        else:
            # Last line fallback
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            pred = lines[-1] if lines else ""

        expected = expected.strip()
        pred = pred.strip()

        # Exact string match
        if pred == expected:
            return 1.0

        # Normalize: remove spaces, $ signs, currency/unit markers, degree symbols
        def norm(s):
            s = s.replace("\\$", "").replace("\\%", "").replace("^\\circ", "")
            s = s.replace("\\circ", "").replace("\\degree", "")
            return re.sub(r"\s+", "", s).replace("$", "").replace(",", "").replace("\\!", "")

        if norm(pred) == norm(expected):
            return 1.0

        # LaTeX-aware normalization: \frac{a}{b} → a/b, \sqrt{x} → sqrt(x)
        def latex_to_expr(s):
            s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
            s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
            s = re.sub(r"\\pi\b", "pi", s)
            s = re.sub(r"\\cdot", "*", s)
            s = s.replace("\\left", "").replace("\\right", "")
            s = s.replace("{", "(").replace("}", ")")
            s = re.sub(r"\s+", "", s)
            return s

        if norm(latex_to_expr(pred)) == norm(latex_to_expr(expected)):
            return 1.0

        # Numeric comparison (no sympy dependency)
        def _try_eval(s):
            """Try to evaluate a math expression to a float."""
            import math
            s = latex_to_expr(s)
            s = s.replace("^", "**")
            # Insert * for implicit multiplication: 2pi → 2*pi, 3sqrt → 3*sqrt
            s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
            s = re.sub(r"(\))(\()", r"\1*\2", s)
            s = re.sub(r"(\d)(\()", r"\1*\2", s)
            # Replace common math constants/functions
            ns = {"pi": math.pi, "sqrt": math.sqrt, "e": math.e,
                  "sin": math.sin, "cos": math.cos, "tan": math.tan,
                  "log": math.log, "ln": math.log}
            try:
                return float(eval(s, {"__builtins__": {}}, ns))
            except Exception:
                return None

        p_val = _try_eval(pred)
        e_val = _try_eval(expected)
        if p_val is not None and e_val is not None:
            if abs(p_val - e_val) < 1e-6:
                return 1.0
            # Relative tolerance for large numbers
            if e_val != 0 and abs(p_val - e_val) / abs(e_val) < 1e-6:
                return 1.0

        return 0.0
