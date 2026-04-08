"""
Hard Code Stream — progressive difficulty code generation benchmark.

Streams code problems at increasing difficulty:
  MBPP (easy) → HumanEval (medium) → CodeContests (hard)

Task format:
  {"id": str, "type": "code_generation"|"code_completion", "domain": str,
   "prompt": str, "answer": str, "test_cases": list, "entry_point": str}

Domains: "mbpp", "humaneval", "code_contests"
"""

import json
import os
import random
import re
import signal
from datasets import load_dataset


HARD_CODE_DOMAINS = ["mbpp", "humaneval", "code_contests"]


def load_hard_code_stream(
    domains: list[str] | None = None,
    n_per_domain: int | dict | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> list[dict]:
    """Load a stream of code problems at increasing difficulty."""
    if domains is None:
        domains = HARD_CODE_DOMAINS

    if n_per_domain is None:
        n_per_domain = {}
    elif isinstance(n_per_domain, int):
        n_per_domain = {d: n_per_domain for d in domains}

    rng = random.Random(seed)
    all_tasks = []

    for domain in domains:
        n = n_per_domain.get(domain, None)
        tasks = _load_code_domain(domain, n, rng, shuffle)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} tasks from {domain}")

    return all_tasks


def _load_code_domain(domain: str, n: int | None, rng: random.Random, shuffle: bool) -> list[dict]:
    tasks = []

    if domain == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        samples = list(ds)
        if shuffle:
            rng.shuffle(samples)
        if n is not None:
            samples = samples[:n]
        for i, item in enumerate(samples):
            test_list = item.get("test_list", [])
            tasks.append({
                "id": f"mbpp_{i}",
                "type": "code_generation",
                "domain": "mbpp",
                "prompt": item["prompt"],
                "answer": item.get("code", ""),
                "test_cases": test_list,
                "entry_point": _extract_entry_point(item.get("code", ""), test_list),
            })

    elif domain == "humaneval":
        ds = load_dataset("evalplus/humanevalplus", split="test")
        samples = list(ds)
        if shuffle:
            rng.shuffle(samples)
        if n is not None:
            samples = samples[:n]
        for i, item in enumerate(samples):
            tasks.append({
                "id": f"humaneval_{i}",
                "type": "code_completion",
                "domain": "humaneval",
                "prompt": item["prompt"],
                "answer": item.get("canonical_solution", ""),
                "test_cases": [item.get("test", "")],
                "entry_point": item.get("entry_point", ""),
            })

    elif domain == "code_contests":
        # Load from local cached parquet (HPC nodes have no internet)
        import glob as _glob
        cache_pattern = os.path.join(
            os.path.expanduser("~"),
            ".cache/huggingface/hub/datasets--deepmind--code_contests/snapshots/*/data/test-*.parquet"
        )
        parquet_files = _glob.glob(cache_pattern)
        if parquet_files:
            ds = load_dataset('parquet', data_files={'test': parquet_files[0]}, split='test')
        else:
            # Fallback: try remote URL (works on login nodes with internet)
            url = ('https://huggingface.co/datasets/deepmind/code_contests/resolve/'
                   'refs%2Fconvert%2Fparquet/default/partial-test/0000.parquet')
            ds = load_dataset('parquet', data_files={'test': url}, split='test')
        samples = list(ds)
        if shuffle:
            rng.shuffle(samples)
        if n is not None:
            samples = samples[:n]
        for i, item in enumerate(samples):
            # Build test cases from public_tests
            pub_tests = item.get("public_tests", {})
            inputs = pub_tests.get("input", []) if pub_tests else []
            outputs = pub_tests.get("output", []) if pub_tests else []
            test_cases = list(zip(inputs, outputs))

            # Also get private tests for evaluation
            priv_tests = item.get("private_tests", {})
            priv_inputs = priv_tests.get("input", []) if priv_tests else []
            priv_outputs = priv_tests.get("output", []) if priv_tests else []
            all_tests = test_cases + list(zip(priv_inputs, priv_outputs))

            # Build prompt from description
            desc = item.get("description", "")
            difficulty = item.get("difficulty", 0)

            tasks.append({
                "id": f"codecontests_{i}",
                "type": "code_generation",
                "domain": "code_contests",
                "prompt": desc,
                "answer": "",  # solutions are in item["solutions"]
                "test_cases": all_tests,  # [(input, expected_output), ...]
                "entry_point": "",  # stdin/stdout problems, no entry point
                "difficulty": difficulty,
                "solutions": item.get("solutions", {}).get("solution", [])[:3],
            })

    return tasks


def _extract_entry_point(code: str, test_list: list) -> str:
    """Extract function name from code or test cases."""
    m = re.search(r"def\s+(\w+)\s*\(", code)
    if m:
        return m.group(1)
    for tc in test_list:
        m = re.search(r"assert\s+(\w+)\s*\(", str(tc))
        if m:
            return m.group(1)
    return ""


class HardCodeEvaluator:
    """Evaluate responses on hard code benchmarks."""

    def __call__(self, task: dict, responses: dict[str, dict]) -> dict:
        scores = {}
        for agent_id, response_dict in responses.items():
            response = response_dict.get("response", "")
            score = self._evaluate(task, response)
            scores[agent_id] = score
        scores["team_score"] = max(scores.values()) if scores else 0.0
        return scores

    def _evaluate(self, task: dict, response: str) -> float:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        code = self._extract_code(response)
        if not code:
            return 0.0

        domain = task.get("domain", "")
        if domain == "code_contests":
            return self._eval_code_contests(code, task)
        elif domain == "humaneval":
            return self._eval_humaneval(code, task)
        else:  # mbpp
            return self._eval_mbpp(code, task)

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            try:
                return response.split("```python")[1].split("```")[0].strip()
            except IndexError:
                pass
        if "```" in response:
            try:
                return response.split("```")[1].split("```")[0].strip()
            except IndexError:
                pass
        if response.strip().startswith(("def ", "import ", "class ", "from ")):
            return response.strip()
        return response.strip()  # for code_contests, the whole response might be code

    def _eval_mbpp(self, code: str, task: dict) -> float:
        test_cases = task.get("test_cases", [])
        if not test_cases:
            return 0.0
        passed = 0
        for tc in test_cases:
            try:
                def _timeout_handler(signum, frame):
                    raise TimeoutError()
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(10)
                try:
                    exec_globals = {}
                    exec(code, exec_globals)
                    exec(tc, exec_globals)
                    passed += 1
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass
        return passed / len(test_cases)

    def _eval_humaneval(self, code: str, task: dict) -> float:
        test_cases = task.get("test_cases", [])
        entry_point = task.get("entry_point", "")
        if not test_cases or not entry_point:
            return 0.0
        try:
            full_code = code + "\n\n" + test_cases[0]
            def _timeout_handler(signum, frame):
                raise TimeoutError()
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(10)
            try:
                exec(full_code, {})
                signal.alarm(0)
                return 1.0
            except Exception:
                signal.alarm(0)
                return 0.0
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            return 0.0

    def _eval_code_contests(self, code: str, task: dict) -> float:
        """Evaluate CodeContests task: stdin/stdout matching."""
        test_cases = task.get("test_cases", [])
        if not test_cases:
            return 0.0

        passed = 0
        total = len(test_cases)

        for inp, expected_out in test_cases:
            try:
                import subprocess
                result = subprocess.run(
                    ["python3", "-c", code],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                actual = result.stdout.strip()
                expected = expected_out.strip()
                if actual == expected:
                    passed += 1
            except Exception:
                pass

        return passed / total if total > 0 else 0.0
