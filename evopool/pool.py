"""
EvoPool: The main pool class that orchestrates everything.

Flow per task:
  1. select_team(task) → k agents
  2. agents execute task
  3. evaluate task result
  4. update individual profiles (individual evolution)
  5. run_codream(team, task, results) (if enabled)
  6. update collab_table
  7. check_and_apply_lifecycle (every LIFECYCLE_CHECK_INTERVAL tasks)
  8. log metrics
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentProfile
from .codream import run_codream, compute_profile_diversity, compute_skill_entropy
from .lifecycle import check_and_apply_lifecycle, LifecycleEvent
from .selector import select_team, CollabScoreTable


LIFECYCLE_CHECK_INTERVAL = 10  # check lifecycle every N tasks


@dataclass
class PoolConfig:
    pool_size_init: int = 20       # starting pool size
    pool_size_min: int = 5
    pool_size_max: int = 50
    team_size: int = 3             # k agents per task
    codream_mode: str = "asymmetric"  # "asymmetric" | "symmetric" | "none"
    mas_mode: str = "leader"          # "leader" | "flat" (flat = old voting, for ablation)
    mas_critique_rounds: int = 1      # 0 = skip critique round in leader mode
    mas_max_extra_agents: int = 2     # cap on GenesisOnDemand recruits per task
    backbone_llm: str = "claude-sonnet-4-6"
    lifecycle_enabled: bool = True
    collab_score_enabled: bool = True
    codream_strength_threshold: float = 0.55
    codream_disable_l3: bool = False     # E22 ablation: disable cross-domain L3 broadcast
    codream_disable_l2: bool = False     # E23 ablation: disable subdomain L2 accumulation
    team_selection_random: bool = False  # E24 ablation: random team selection
    codream_enhanced: bool = False       # E25: disagreement trigger + success extraction + domain_general
    codream_no_verify: bool = False      # E27: skip verify gate, apply all insights directly
    seed: int = 42


class EvoPool:
    """
    The main EvoPool class.
    Manages a pool of evolving agents across a lifelong task stream.
    """

    def __init__(self, config: PoolConfig):
        self.config = config
        self.pool: list[Agent] = self._init_pool()
        self.collab_table = CollabScoreTable()
        self.task_index: int = 0
        self.recent_task_types: list[str] = []
        self.recent_tasks: list[dict] = []  # full task dicts for embedding-based detection
        self.lifecycle_events: list[LifecycleEvent] = []
        self.metrics_log: list[dict] = []

    # Diverse initial personas to bootstrap specialization.
    # One-third math/science, one-third code/engineering, one-third QA/language.
    _INIT_PERSONAS = [
        # Math / reasoning specialists
        ("math_specialist",
         "A mathematical reasoning expert skilled in algebra, calculus, combinatorics, and "
         "competition mathematics. Strong at step-by-step derivations and numerical verification.",
         {"math_competition": 0.7, "math_word_problem": 0.65, "arithmetic": 0.7}),
        ("quantitative_analyst",
         "An analytical agent with deep experience in quantitative problem solving, "
         "statistical reasoning, and structured multi-step arithmetic.",
         {"math_word_problem": 0.7, "math_competition": 0.6}),
        ("logic_reasoner",
         "A deductive reasoning specialist adept at formal logic, proofs, and "
         "multi-step inference chains.",
         {"math_competition": 0.6, "multi_hop_qa": 0.6}),
        # Code / engineering specialists
        ("python_engineer",
         "A senior Python engineer with deep expertise in algorithms, data structures, "
         "and writing correct, efficient code solutions with passing test cases.",
         {"code_generation": 0.75, "code_completion": 0.75, "programming": 0.7}),
        ("software_developer",
         "A software development expert skilled at implementing functions from specifications, "
         "debugging, and ensuring code correctness against test suites.",
         {"code_generation": 0.7, "code_completion": 0.7}),
        ("algorithm_specialist",
         "An algorithms and data structures expert who writes clean, optimized Python code "
         "and understands time/space complexity trade-offs.",
         {"code_completion": 0.7, "code_generation": 0.65}),
        # QA / language specialists
        ("qa_researcher",
         "A research-oriented question-answering agent skilled at multi-hop reasoning, "
         "evidence synthesis, and extracting precise answers from passages.",
         {"multi_hop_qa": 0.75, "reading_comprehension": 0.7, "factual_qa": 0.7}),
        ("reading_comprehension_expert",
         "An expert at reading comprehension and information extraction from complex passages, "
         "including numerical and span-based answers.",
         {"reading_comprehension": 0.75, "multi_hop_qa": 0.65}),
    ]

    def _init_pool(self) -> list[Agent]:
        """
        Initialize pool with a mix of domain-specialist and generalist agents.
        The first 8 agents get domain-specific priors; the rest are generalists.
        This bootstraps specialization so team selection is meaningful from task 1.
        """
        agents = []
        n = self.config.pool_size_init
        n_spec = min(len(self._INIT_PERSONAS), n)

        for i in range(n_spec):
            agent_tag, persona, skill_prior = self._INIT_PERSONAS[i]
            profile = AgentProfile(
                persona=f"Agent-{i} ({agent_tag}): {persona}",
                skill_memory=dict(skill_prior),
                task_history=[],
                collab_log={},
                perf_stats={},
            )
            agents.append(Agent(profile=profile))

        for i in range(n_spec, n):
            profile = AgentProfile(
                persona=f"Agent-{i}: A versatile general-purpose AI assistant with broad capabilities.",
                skill_memory={},
                task_history=[],
                collab_log={},
                perf_stats={},
            )
            agents.append(Agent(profile=profile))
        return agents

    # ------------------------------------------------------------------
    # Main task processing loop
    # ------------------------------------------------------------------

    def process_task(self, task: dict, evaluator) -> dict:
        """
        Process a single task through the full EvoPool pipeline.

        Args:
            task: dict with at least {"id", "type", "prompt"}
            evaluator: callable(task, responses) -> {agent_id: score, "team_score": float}

        Returns:
            result dict with scores, metrics, lifecycle events
        """
        # 1. Select team
        if self.config.team_selection_random:
            import random as _random
            team = _random.sample(self.pool, min(self.config.team_size, len(self.pool)))
        else:
            team = select_team(
                pool=self.pool,
                task=task,
                collab_table=self.collab_table,
                k=self.config.team_size,
            )

        # 2. Execute task via MAS structure
        leader_id = None
        extra_recruited: list[str] = []
        decomposition_plan = None

        if self.config.mas_mode == "leader":
            from .mas import run_leader_mas
            mas_result = run_leader_mas(
                team=team,
                task=task,
                pool=self.pool,
                backbone_llm=self.config.backbone_llm,
                max_extra_agents=self.config.mas_max_extra_agents,
                critique_enabled=(self.config.mas_critique_rounds > 0),
            )
            responses = mas_result.per_agent_responses
            leader_id = mas_result.leader_id
            extra_recruited = mas_result.extra_agents_recruited
            decomposition_plan = mas_result.decomposition_plan
            _leader_feedback = mas_result.per_agent_feedback
            # Add recruited agents to team for profile updates and Co-Dream
            recruited_agents = [a for a in self.pool if a.agent_id in extra_recruited]
            team = team + recruited_agents
            # Include synthesized answer in evaluation (may be better than individual responses)
            if mas_result.final_answer:
                responses["__synthesized__"] = {
                    "agent_id": "__synthesized__",
                    "response": mas_result.final_answer,
                    "task_type": task.get("type", "unknown"),
                }
        else:
            # flat mode: each agent independently answers
            responses = {}
            for agent in team:
                responses[agent.agent_id] = agent.execute_task(task, self.config.backbone_llm)
            _leader_feedback = {}

        # 3. Evaluate
        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.5)
        pool_mean = self._compute_pool_mean_score()

        # Blend evaluator score with leader's process feedback (leader mode only)
        if self.config.mas_mode == "leader" and _leader_feedback:
            for agent in team:
                aid = agent.agent_id
                eval_score = evaluation.get(aid, team_score)
                leader_hint = _leader_feedback.get(aid, {}).get("score_hint", eval_score)
                evaluation[aid] = 0.7 * eval_score + 0.3 * float(leader_hint)

        # 4. Individual profile update
        for agent in team:
            agent_score = evaluation.get(agent.agent_id, team_score)
            outcome = {
                "score": agent_score,
                "label": "correct" if agent_score > 0.5 else "incorrect",
                "pool_mean_score": pool_mean,
            }
            agent.update_from_feedback(task, outcome, self.config.backbone_llm)

        # 5. Co-Dream (5-phase: reflect → contrast → imagine → debate → crystallize)
        # Build a lightweight single-response evaluator for insight verification.
        # evaluator(task, response_str) -> float  (not counted in benchmark)
        def _single_response_eval(t, response_str):
            try:
                result = evaluator(t, {a.agent_id: response_str for a in team[:1]})
                return result.get("team_score", 0.0)
            except Exception:
                return 0.0

        # E27: no-verify mode skips the verify gate (evaluator_fn=None → apply all insights)
        use_evaluator = (
            self.config.codream_mode != "none"
            and not self.config.codream_no_verify
        )
        codream_session = run_codream(
            team=team,
            task=task,
            task_results={a.agent_id: {"score": evaluation.get(a.agent_id, team_score)} for a in team},
            backbone_llm=self.config.backbone_llm,
            mode=self.config.codream_mode,
            strength_threshold=self.config.codream_strength_threshold,
            evaluator_fn=_single_response_eval if use_evaluator else None,
            disable_l3=self.config.codream_disable_l3,
            disable_l2=self.config.codream_disable_l2,
            enhanced=self.config.codream_enhanced,
        )

        # 6. Update collab table
        if self.config.collab_score_enabled:
            self.collab_table.record_team_result(team, team_score)

        # 7. Track task type and full task for embedding-based domain detection
        task_type = task.get("type", "general")
        self.recent_task_types.append(task_type)
        self.recent_tasks.append(task)
        if len(self.recent_tasks) > 20:
            self.recent_tasks = self.recent_tasks[-20:]

        # 8. Lifecycle check — with semantic domain-shift detection
        lifecycle_events_this_task = []
        if self.config.lifecycle_enabled and self.task_index % LIFECYCLE_CHECK_INTERVAL == 0:
            from .task_embed import is_domain_shift
            domain_shift = is_domain_shift(
                task, self.recent_tasks[:-1], threshold=0.65, min_recent=3
            ) if len(self.recent_tasks) > 3 else False

            self.pool, lifecycle_events_this_task = check_and_apply_lifecycle(
                pool=self.pool,
                task_index=self.task_index,
                recent_task_types=self.recent_task_types,
                pool_mean_score=pool_mean,
                backbone_llm=self.config.backbone_llm,
                n_min=self.config.pool_size_min,
                n_max=self.config.pool_size_max,
                current_task_type=task_type,
                current_task=task,
                recent_tasks=self.recent_tasks[:-1],
                domain_shift=domain_shift,
            )
            self.lifecycle_events.extend(lifecycle_events_this_task)

        # 9. Log metrics
        metrics = self._compute_metrics(team_score, lifecycle_events_this_task)
        self.metrics_log.append(metrics)

        self.task_index += 1

        cd = codream_session
        result_dict: dict = {
            "task_id": task.get("id", self.task_index),
            "team_score": team_score,
            "team_agent_ids": [a.agent_id for a in team],
            "leader_agent_id": leader_id,
            "extra_agents_recruited": extra_recruited,
            "decomposition_plan": decomposition_plan,
            "codream_insights": len(cd.insights) if cd else 0,
            "codream_generated": cd.n_insights_generated if cd else 0,
            "codream_verified": cd.n_insights_verified if cd else 0,
            "codream_verify_rate": cd.verify_rate if cd else 0.0,
            "lifecycle_events": [e.__dict__ for e in lifecycle_events_this_task],
            "metrics": metrics,
        }
        # Include final_answer for code tasks (aids debugging)
        if self.config.mas_mode == "leader" and task.get("domain") in ("humaneval", "mbpp"):
            result_dict["final_answer"] = mas_result.final_answer if mas_result else ""
        return result_dict

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_pool_mean_score(self) -> float:
        """Compute mean performance across all recent tasks and agents."""
        all_scores = []
        for agent in self.pool:
            for domain, scores in agent.profile.perf_stats.items():
                all_scores.extend(scores[-5:])
        return sum(all_scores) / len(all_scores) if all_scores else 0.5

    def _compute_metrics(self, team_score: float, lifecycle_events: list) -> dict:
        return {
            "task_index": self.task_index,
            "team_score": team_score,
            "pool_size": len(self.pool),
            "profile_diversity": compute_profile_diversity(self.pool),
            "skill_entropy": compute_skill_entropy(self.pool),
            "n_lifecycle_events": len(lifecycle_events),
            "timestamp": time.time(),
        }

    def get_metrics_df(self):
        """Return metrics as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.metrics_log)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "config": self.config.__dict__,
            "pool": [a.to_dict() for a in self.pool],
            "collab_table": self.collab_table.to_dict(),
            "task_index": self.task_index,
            "recent_task_types": self.recent_task_types[-100:],
            "lifecycle_events": [e.__dict__ for e in self.lifecycle_events],
            "metrics_log": self.metrics_log,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> EvoPool:
        with open(path) as f:
            state = json.load(f)
        config = PoolConfig(**state["config"])
        pool_obj = cls(config)
        pool_obj.pool = [Agent.from_dict(d) for d in state["pool"]]
        pool_obj.collab_table = CollabScoreTable.from_dict(state["collab_table"])
        pool_obj.task_index = state["task_index"]
        pool_obj.recent_task_types = state["recent_task_types"]
        pool_obj.metrics_log = state["metrics_log"]
        return pool_obj
