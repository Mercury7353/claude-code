"""
EvoMem Baseline: Individual Evolving Memory (Reflexion-style).

Each agent maintains its own private memory that evolves through self-reflection
after each task. Unlike CoDream, there is NO cross-agent knowledge sharing.

Purpose: Isolate the contribution of individual memory evolution vs. collective
memory evolution (CoDream). If EvoPool >> EvoMem, the gain comes from collective
knowledge sharing, not just individual memory accumulation.

Design:
  - Pool of N agents, same team selection as EvoPool (affinity + diversity)
  - After each task: agents that FAILED generate a self-reflection:
      "I failed this task. What went wrong? What should I try differently?"
  - Reflection stored in agent's private memory (not shared with anyone)
  - At inference time: agent reads its own past reflections, NOT others'
  - NO crystallize/broadcast step (that's CoDream)

Key contrast with:
  - AgentNet: retrieves past task snippets; EvoMem reflects on WHY it failed
  - MemCollab: shared memory; EvoMem is strictly private
  - EvoPool-noCoDream (E15b): has profile history but no active self-reflection LLM call
  - EvoPool-full: CoDream broadcasts winning agent's insights to losers (asymmetric)
"""

from __future__ import annotations

import random
from typing import Optional

from evopool.llm import llm_call


class EvoMemPool:
    """
    Pool with individual self-reflection memory (no cross-agent sharing).
    """

    def __init__(
        self,
        pool_size: int = 20,
        team_size: int = 3,
        backbone_llm: str = "qwen3-8b",
        seed: int = 42,
        memory_per_agent: int = 10,
    ):
        self.pool_size = pool_size
        self.team_size = team_size
        self.backbone_llm = backbone_llm
        self.rng = random.Random(seed)
        self.memory_per_agent = memory_per_agent

        self.task_index = 0
        self.metrics_log: list[dict] = []

        # Per-agent private memory: {agent_id: [reflection_str, ...]}
        self.agents: list[dict] = [
            {
                "id": f"evomem_{i}",
                "reflections": [],       # private, never shared
                "scores": [],            # task score history
                "domains": [],           # domain history
                "collab_scores": {},     # agent_id -> float (affinity for team selection)
            }
            for i in range(pool_size)
        ]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def process_task(self, task: dict, evaluator) -> dict:
        domain = task.get("domain", "general")
        task_type = task.get("type", "general")
        task_prompt = task.get("prompt", str(task))

        # Select team (affinity + diversity, same as EvoPool)
        team = self._select_team(domain)

        # Generate responses with private memory injection
        responses: dict[str, dict] = {}
        for agent in team:
            prompt = self._build_prompt(agent, task_prompt, domain, task_type)
            response = llm_call(
                model=self.backbone_llm,
                system=f"You are agent {agent['id']}, a problem-solving specialist.",
                user=prompt,
            )
            responses[agent["id"]] = {
                "agent_id": agent["id"],
                "response": response,
                "task_type": task_type,
            }

        # Evaluate
        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.0)

        # Individual self-reflection update (private, no sharing)
        self._individual_reflect(team, task, responses, evaluation, domain)

        # Update agent stats
        for agent in team:
            agent_score = evaluation.get(agent["id"], team_score)
            agent["scores"].append(agent_score)
            agent["domains"].append(domain)

        # Update pairwise collab scores
        self._update_collab_scores(team, team_score)

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
            "team_agent_ids": [a["id"] for a in team],
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Team selection: affinity + diversity (mirrors EvoPool)
    # ------------------------------------------------------------------

    def _select_team(self, domain: str) -> list[dict]:
        """Select team_size agents by domain affinity + diversity."""
        def domain_score(agent) -> float:
            domain_tasks = [s for s, d in zip(agent["scores"], agent["domains"]) if d == domain]
            if not domain_tasks:
                return 0.5  # prior
            return sum(domain_tasks[-10:]) / len(domain_tasks[-10:])

        # Sort by domain affinity
        sorted_agents = sorted(self.agents, key=domain_score, reverse=True)

        # Take top-2 by affinity, then add 1 random for diversity
        team = sorted_agents[:2]
        remaining = [a for a in sorted_agents[2:] if a not in team]
        if remaining:
            team.append(self.rng.choice(remaining))
        elif len(team) < self.team_size:
            team.append(sorted_agents[0])  # fallback

        return team[:self.team_size]

    # ------------------------------------------------------------------
    # Prompt construction: inject private reflections
    # ------------------------------------------------------------------

    def _build_prompt(self, agent: dict, task_prompt: str, domain: str, task_type: str) -> str:
        # Retrieve domain-relevant reflections (most recent first)
        relevant = [r for r in reversed(agent["reflections"]) if r.get("domain") == domain]
        recent_general = [r for r in reversed(agent["reflections"])]
        # Prefer domain-specific, fill with general
        picks = relevant[:2] + [r for r in recent_general if r not in relevant][:1]

        if not picks:
            return task_prompt

        mem_lines = "\n".join(f"- {r['text']}" for r in picks[:3])
        return (
            f"[Your private reflections from past tasks in this domain]\n{mem_lines}\n\n"
            f"Apply these lessons to the current task:\n{task_prompt}"
        )

    # ------------------------------------------------------------------
    # Individual self-reflection (the core EvoMem mechanism)
    # ------------------------------------------------------------------

    def _individual_reflect(
        self,
        team: list[dict],
        task: dict,
        responses: dict[str, dict],
        evaluation: dict,
        domain: str,
    ) -> None:
        """
        Each agent that failed reflects privately on what went wrong.
        Reflection is stored only in that agent's private memory.
        """
        task_prompt = task.get("prompt", "")[:300]
        best_score = evaluation.get("team_score", 0.0)

        for agent in team:
            agent_score = evaluation.get(agent["id"], best_score)
            response_text = responses.get(agent["id"], {}).get("response", "")

            # Only reflect on failures (score < 0.5)
            if agent_score >= 0.5:
                continue

            try:
                reflection_text = llm_call(
                    model=self.backbone_llm,
                    system="You are reflecting on a failed task to improve future performance.",
                    user=(
                        f"You attempted this {domain} task and got it wrong.\n\n"
                        f"Task: {task_prompt}\n\n"
                        f"Your answer: {response_text[:200]}\n\n"
                        f"Write 1-2 sentences: what specifically went wrong, "
                        f"and what strategy should you use differently next time? "
                        f"Be concrete and generalizable to similar problems."
                    ),
                )
            except Exception:
                continue

            if not reflection_text or len(reflection_text) < 15:
                continue

            agent["reflections"].append({
                "domain": domain,
                "text": reflection_text[:300],
                "task_index": self.task_index,
            })
            # Keep only recent memory
            agent["reflections"] = agent["reflections"][-self.memory_per_agent:]

    # ------------------------------------------------------------------
    # Collab score update
    # ------------------------------------------------------------------

    def _update_collab_scores(self, team: list[dict], team_score: float) -> None:
        for i, a in enumerate(team):
            for j, b in enumerate(team):
                if i == j:
                    continue
                prev = a["collab_scores"].get(b["id"], 0.5)
                a["collab_scores"][b["id"]] = 0.8 * prev + 0.2 * team_score
