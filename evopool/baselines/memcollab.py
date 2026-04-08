"""
MemCollab Baseline: Contrastive Trajectory Distillation for Multi-Agent Memory.
arXiv: 2603.23234

Key idea: After each task, distill successful vs failed agent trajectories into a
shared, flat collective memory. Future agents retrieve this shared memory for
retrieval-augmented generation (RAG). Unlike per-agent RAG (AgentNet), MemCollab
builds ONE shared memory store accessed by all agents.

Contrast with EvoPool CoDream:
  - MemCollab: symmetric distillation, flat shared memory, contrastive training signal
  - EvoPool CoDream: asymmetric (successful->failing), 3-tier hierarchical memory,
    verify gate for insight quality, LLM-mediated insight crystallization

Implementation faithful to arXiv 2603.23234:
  1. Multi-agent answer generation (k agents)
  2. Contrastive memory update: distill (success, fail) pairs into shared pool
  3. Memory-augmented prompting: retrieve top-k relevant traces
  4. Periodic consolidation: merge redundant memories via LLM
"""

from __future__ import annotations

import random
from typing import Optional

from evopool.llm import llm_call


class MemCollabPool:
    """
    MemCollab: shared contrastive memory across all agents.
    """

    def __init__(
        self,
        pool_size: int = 10,
        team_size: int = 3,
        backbone_llm: str = "qwen3-8b",
        seed: int = 42,
        memory_capacity: int = 200,
        retrieval_k: int = 3,
        consolidation_interval: int = 50,
    ):
        self.pool_size = pool_size
        self.team_size = team_size
        self.backbone_llm = backbone_llm
        self.rng = random.Random(seed)
        self.memory_capacity = memory_capacity
        self.retrieval_k = retrieval_k
        self.consolidation_interval = consolidation_interval

        self.task_index = 0
        self.metrics_log: list[dict] = []

        # Shared collective memory: list of memory entries
        # Each entry: {"domain": str, "task_snippet": str, "approach": str,
        #               "outcome": "success"|"failure", "score": float}
        self.shared_memory: list[dict] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def process_task(self, task: dict, evaluator) -> dict:
        """Run MemCollab: retrieve → generate → contrastive update."""
        task_prompt = task.get("prompt", str(task))
        domain = task.get("domain", "general")
        task_type = task.get("type", "general")

        # For code tasks, include function name and test cases
        if domain in ("mbpp", "humaneval") or task_type in ("code_generation", "code_completion"):
            import re as _re
            entry_point = task.get("entry_point", "")
            if not entry_point:
                for tc in (task.get("test_cases") or []):
                    _m = _re.search(r"assert\s+(\w+)\s*\(", str(tc))
                    if _m:
                        entry_point = _m.group(1)
                        break
            if entry_point:
                task_prompt = f"[REQUIRED FUNCTION NAME: {entry_point}]\n\n" + task_prompt
            test_cases = task.get("test_cases", [])
            if test_cases:
                task_prompt += "\n\nTest cases:\n" + "\n".join(str(tc) for tc in test_cases[:3])

        is_hard_math = (
            task.get("type") in ("aime_problem", "math_competition_hard")
            or domain.startswith("aime_")
            or domain == "math_hard"
        )

        # Step 1: Retrieve relevant memories from shared pool
        retrieved = self._retrieve(domain, task_prompt, k=self.retrieval_k)

        # Step 2: Multi-agent generation with shared memory context
        responses: dict[str, dict] = {}
        agent_ids = [f"memcollab_{i}" for i in range(self.team_size)]
        for agent_id in agent_ids:
            prompt = self._build_prompt(task_prompt, retrieved, task_type)
            response = llm_call(
                model=self.backbone_llm,
                system=f"You are a problem-solving agent with access to shared team memory.",
                user=prompt,
                max_tokens=4096 if is_hard_math else 512,
                enable_thinking=is_hard_math,
            )
            responses[agent_id] = {
                "agent_id": agent_id,
                "response": response,
                "task_type": task_type,
            }

        # Step 3: Evaluate
        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.0)

        # Step 4: Contrastive memory update
        self._contrastive_update(task, responses, evaluation, domain, task_prompt)

        # Step 5: Periodic consolidation
        if self.task_index > 0 and self.task_index % self.consolidation_interval == 0:
            self._consolidate_memory(domain)

        self.task_index += 1
        metrics = {
            "task_index": self.task_index,
            "team_score": team_score,
            "memory_size": len(self.shared_memory),
            "pool_size": self.pool_size,
            "profile_diversity": 0.0,
        }
        self.metrics_log.append(metrics)

        return {
            "task_id": task.get("id", self.task_index),
            "team_score": team_score,
            "team_agent_ids": agent_ids,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Retrieval: keyword overlap + domain matching
    # ------------------------------------------------------------------

    def _retrieve(self, domain: str, task_prompt: str, k: int) -> list[dict]:
        """Retrieve top-k relevant memories from shared pool."""
        if not self.shared_memory:
            return []

        task_words = set(task_prompt.lower().split())

        def score_entry(entry: dict) -> float:
            s = 0.0
            # Domain match
            if entry.get("domain") == domain:
                s += 2.0
            # Keyword overlap with task snippet
            entry_words = set(entry.get("task_snippet", "").lower().split())
            overlap = len(task_words & entry_words)
            s += overlap * 0.1
            # Prefer successful memories slightly
            if entry.get("outcome") == "success":
                s += 0.5
            return s

        scored = sorted(self.shared_memory, key=score_entry, reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        task_prompt: str,
        retrieved: list[dict],
        task_type: str,
    ) -> str:
        if not retrieved:
            return task_prompt

        mem_lines = []
        for i, mem in enumerate(retrieved):
            outcome_tag = "[SUCCESS]" if mem["outcome"] == "success" else "[FAILURE - avoid]"
            mem_lines.append(
                f"Memory {i+1} {outcome_tag}: {mem['task_snippet'][:100]}...\n"
                f"  Approach: {mem['approach'][:200]}"
            )

        mem_block = "\n".join(mem_lines)
        return (
            f"=== Shared Team Memory ===\n{mem_block}\n\n"
            f"=== Current Task ===\n{task_prompt}"
        )

    # ------------------------------------------------------------------
    # Contrastive memory update (core MemCollab mechanism)
    # ------------------------------------------------------------------

    def _contrastive_update(
        self,
        task: dict,
        responses: dict[str, dict],
        evaluation: dict,
        domain: str,
        task_prompt: str,
    ) -> None:
        """
        Update shared memory with success/failure pairs.

        For each agent, record its approach with outcome label.
        Contrastive signal: the shared memory stores both successes (to imitate)
        and failures (to avoid), enabling contrastive learning at inference time.
        """
        task_snippet = task_prompt[:150]

        for agent_id, resp_dict in responses.items():
            agent_score = evaluation.get(agent_id, evaluation.get("team_score", 0.0))
            response_text = resp_dict.get("response", "")
            if not response_text:
                continue

            outcome = "success" if agent_score >= 0.5 else "failure"

            # Extract compact approach description
            approach = self._extract_approach(response_text, task.get("type", "general"))

            entry = {
                "domain": domain,
                "task_snippet": task_snippet,
                "approach": approach,
                "outcome": outcome,
                "score": agent_score,
                "task_index": self.task_index,
            }
            self.shared_memory.append(entry)

        # Enforce capacity limit: keep most recent + highest-scoring entries
        if len(self.shared_memory) > self.memory_capacity:
            # Sort: keep successes first, then by recency
            successes = [m for m in self.shared_memory if m["outcome"] == "success"]
            failures = [m for m in self.shared_memory if m["outcome"] == "failure"]
            # Keep recent successes and a sample of failures for contrastive signal
            successes = successes[-int(self.memory_capacity * 0.7):]
            failures = failures[-int(self.memory_capacity * 0.3):]
            self.shared_memory = successes + failures

    def _extract_approach(self, response: str, task_type: str) -> str:
        """Extract a compact approach description from agent response."""
        # Take first 300 chars (the "approach" part, before detailed computation)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if not lines:
            return response[:300]
        # For math/code: return first substantive line as approach
        approach_lines = []
        for line in lines[:5]:
            if len(line) > 20:
                approach_lines.append(line)
        return " | ".join(approach_lines[:3]) if approach_lines else lines[0][:300]

    # ------------------------------------------------------------------
    # Periodic consolidation: merge redundant memories via LLM
    # ------------------------------------------------------------------

    def _consolidate_memory(self, domain: str) -> None:
        """
        Consolidate domain-specific memories: merge similar success entries
        into generalized strategy strings. This is MemCollab's distillation step.
        """
        domain_memories = [m for m in self.shared_memory if m["domain"] == domain and m["outcome"] == "success"]
        if len(domain_memories) < 5:
            return

        # Sample up to 10 recent successes
        sample = domain_memories[-10:]
        snippets = "\n".join(
            f"- Approach: {m['approach'][:150]}" for m in sample
        )

        try:
            consolidated = llm_call(
                model=self.backbone_llm,
                system="You are a memory consolidation agent.",
                user=(
                    f"Synthesize these successful problem-solving approaches for {domain} tasks "
                    f"into 1-2 general strategies:\n{snippets}\n\n"
                    f"Output: A concise strategy (2-3 sentences max)."
                ),
            )
        except Exception:
            return

        if not consolidated or len(consolidated) < 20:
            return

        # Replace sampled entries with one consolidated entry
        for m in sample:
            if m in self.shared_memory:
                self.shared_memory.remove(m)

        self.shared_memory.append({
            "domain": domain,
            "task_snippet": f"[consolidated from {len(sample)} tasks]",
            "approach": consolidated[:400],
            "outcome": "success",
            "score": 1.0,
            "task_index": self.task_index,
        })
