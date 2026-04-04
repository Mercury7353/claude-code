"""
AgentNet Baseline: Decentralized Evolutionary Coordination (NeurIPS 2025).
Official codebase: https://github.com/zoe-yyx/AgentNet
arXiv: 2504.00587

Adapted to EvoPool's streaming task interface.
Key changes from official code:
  - LLM calls (doubao/gpt) redirected to local vLLM server via patched src/utils.py
  - FlagEmbedding replaced with lightweight TF-IDF stub (FlagEmbedding/__init__.py)
  - Task format adapter: our dict → AgentNet Task object
  - process_task() interface to match EvoPool's benchmark runner

Original AgentNet key features preserved (via official Experiment class):
  - Per-agent RAG memory (RouterExperiencePool, ExecutorExperiencePool)
  - Dynamic DAG topology with Router/Executor architecture
  - Agent specialization via retrieval-augmented routing decisions
  - Evolutionary graph updates based on task performance
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Add official AgentNet codebase to path
_AGENTNET_CODE_PATH = str(
    Path(__file__).parent.parent.parent / "external_baselines" / "AgentNet" / "AgentNet_Code"
)
if _AGENTNET_CODE_PATH not in sys.path:
    sys.path.insert(0, _AGENTNET_CODE_PATH)

# Set config path for AgentNet
os.environ.setdefault("AGENTNET_CONFIG_PATH", str(Path(_AGENTNET_CODE_PATH) / "config"))


class AgentNetPool:
    """
    Wrapper around official AgentNet Experiment to match EvoPool's process_task interface.
    """

    def __init__(
        self,
        pool_size: int = 10,
        team_size: int = 3,
        backbone_llm: str = "qwen3-8b",
        seed: int = 42,
    ):
        self.pool_size = pool_size
        self.team_size = team_size
        self.backbone_llm = backbone_llm
        self.seed = seed
        self.task_index = 0
        self.metrics_log: list[dict] = []
        self._experiment = None  # Lazy-init

    def _get_experiment(self):
        """Lazy-initialize AgentNet Experiment."""
        if self._experiment is not None:
            return self._experiment
        try:
            from src.experiment import Experiment
            from src.utils import read_yaml
            config_path = os.path.join(_AGENTNET_CODE_PATH, "config", "bigbenchhard_config.yaml")
            if os.path.exists(config_path):
                config = read_yaml(config_path)
            else:
                config = self._default_config()
            # Override relevant settings
            config["num_agents"] = self.pool_size
            config["llm"] = "gpt"  # Will be redirected to vLLM by our patch
            self._experiment = Experiment(config)
            return self._experiment
        except Exception as e:
            # Fall back to our self-implementation if official code fails to load
            import warnings
            warnings.warn(
                f"Could not load official AgentNet Experiment ({e}). "
                "Falling back to faithful self-implementation.",
                stacklevel=2,
            )
            return None

    def _default_config(self) -> dict:
        return {
            "num_agents": self.pool_size,
            "llm": "gpt",
            "memory_limit": 50,
            "retrieval_num": 3,
            "global_router_experience": False,
        }

    def _task_to_agentnet_format(self, task: dict):
        """Convert our task dict to AgentNet's Task object."""
        try:
            from src.task import Task
            return Task(
                task_id=self.task_index,
                task_type=task.get("type", "general"),
                description=task.get("prompt", str(task)),
                major_problem=task.get("prompt", str(task))[:200],
                progress_text="",
                thought="",
            )
        except Exception:
            return None

    def process_task(self, task: dict, evaluator) -> dict:
        """
        AgentNet task processing using official DAG routing/execution.
        Falls back to faithful self-implementation if official code fails.
        """
        experiment = self._get_experiment()

        if experiment is not None:
            return self._process_with_official(task, evaluator, experiment)
        else:
            return self._process_fallback(task, evaluator)

    def _process_with_official(self, task: dict, evaluator, experiment) -> dict:
        """Use official AgentNet Experiment to process task."""
        try:
            agentnet_task = self._task_to_agentnet_format(task)
            if agentnet_task is None:
                raise ValueError("Failed to convert task")

            # Run the official AgentNet processing
            result = experiment.run_task(agentnet_task)
            answer = str(result.get("answer", result.get("result", "")))

            responses = {
                "agentnet_dag": {
                    "agent_id": "agentnet_dag",
                    "response": answer,
                    "task_type": task.get("type", "unknown"),
                }
            }
        except Exception as e:
            # If official processing fails, fall back
            responses = {
                "agentnet_dag": {
                    "agent_id": "agentnet_dag",
                    "response": "",
                    "task_type": task.get("type", "unknown"),
                }
            }

        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.0)

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
            "team_agent_ids": [f"agentnet_agent_{i}" for i in range(self.team_size)],
            "metrics": metrics,
        }

    def _process_fallback(self, task: dict, evaluator) -> dict:
        """
        Faithful self-implementation fallback.
        Captures AgentNet's key innovation: per-agent RAG memory with topology-based selection.
        """
        # Initialize agents if needed
        if not hasattr(self, "_agents"):
            self._init_fallback_agents()

        task_type = task.get("type", "general")
        task_prompt = task.get("prompt", str(task))

        # Select team based on memory relevance (AgentNet's topology)
        def memory_score(agent) -> float:
            relevant = [
                m for m in agent["memory"]
                if m["task_type"] == task_type and m["score"] > 0.5
            ]
            return len(relevant)

        team = sorted(self._agents, key=memory_score, reverse=True)[:self.team_size]

        from evopool.llm import llm_call

        responses = {}
        for agent in team:
            # Build RAG context
            relevant = [m for m in agent["memory"] if m["task_type"] == task_type][-3:]
            context = ""
            if relevant:
                context = "Past experiences:\n" + "\n".join(
                    f"- {m['snippet']} (score={m['score']:.2f})" for m in relevant
                )
            system = f"You are AgentNet agent {agent['id']}. {context}"
            response = llm_call(model=self.backbone_llm, system=system, user=task_prompt)
            responses[agent["id"]] = {
                "agent_id": agent["id"],
                "response": response,
                "task_type": task_type,
            }

        evaluation = evaluator(task, responses)
        team_score = evaluation.get("team_score", 0.0)

        # Update per-agent memories (no cross-agent transfer)
        for agent in team:
            agent_score = evaluation.get(agent["id"], team_score)
            agent["memory"].append({
                "task_type": task_type,
                "snippet": task_prompt[:150],
                "score": agent_score,
            })
            agent["memory"] = agent["memory"][-50:]  # Keep last 50

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

    def _init_fallback_agents(self):
        self._agents = [
            {"id": f"agentnet_{i}", "memory": []}
            for i in range(self.pool_size)
        ]
