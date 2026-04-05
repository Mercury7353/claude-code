#!/usr/bin/env python3
"""
Main experiment runner for EvoPool.

Usage:
  python run_experiment.py --condition evopool_full --benchmark aflow_stream --seed 42
  python run_experiment.py --condition dylan --benchmark aflow_stream --seed 42
  python run_experiment.py --condition agentnet --benchmark aflow_stream --seed 42
  python run_experiment.py --condition evopool_no_codream --benchmark aflow_stream --seed 42
  python run_experiment.py --condition evopool_symmetric_codream --benchmark aflow_stream --seed 42

Available conditions:
  evopool_full            EvoPool with all components
  evopool_no_codream      EvoPool without Co-Dream (individual memory only)
  evopool_symmetric_cod   EvoPool with symmetric Co-Dream (ablation)
  evopool_no_lifecycle    EvoPool without lifecycle operators
  evopool_no_collab_score EvoPool without historical collab score
  dylan                   DyLAN baseline (static pool + AIS)
  agentnet                AgentNet baseline (per-agent RAG memory)
  no_memory               No memory baseline (random selection, no profile update)
  self_consistency        Self-Consistency (Wang et al. 2022): k=5 majority vote, no memory

Available benchmarks:
  aflow_stream     6-domain sequential stream (GSM8K/HotpotQA/MBPP/MATH/HumanEval/DROP)
  gsm8k_stream     GSM8K only (for pilot experiments)
"""

import argparse
import json
import os
import sys
import time
import traceback

from evopool.benchmarks.aflow_stream import load_aflow_stream, AFlowEvaluator
from evopool.eval.metrics import summarize_results, print_comparison_table


def main():
    parser = argparse.ArgumentParser(description="Run EvoPool experiment")
    parser.add_argument("--condition", type=str, required=True, help="System to evaluate")
    parser.add_argument("--benchmark", type=str, default="aflow_stream")
    parser.add_argument("--n_tasks", type=int, default=60, help="Total tasks in stream")
    parser.add_argument("--n_per_domain", type=int, default=10, help="Tasks per domain")
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--team_size", type=int, default=3)
    parser.add_argument("--backbone_llm", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domain list (default: all 6)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"{args.condition}_{args.benchmark}_seed{args.seed}.json"
    )

    print(f"=== EvoPool Experiment ===")
    print(f"Condition: {args.condition}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Seed: {args.seed}")
    print(f"Pool size: {args.pool_size}, Team size: {args.team_size}")
    print(f"Backbone LLM: {args.backbone_llm}")
    print(f"Output: {output_file}")
    print()

    # Load benchmark
    domains = args.domains.split(",") if args.domains else None
    if args.benchmark == "gsm8k_stream":
        domains = ["gsm8k"]
        n_per_domain = args.n_tasks
    else:
        n_per_domain = args.n_per_domain

    print(f"Loading tasks (benchmark={args.benchmark}, n_per_domain={n_per_domain})...")
    tasks = load_aflow_stream(
        n_per_domain=n_per_domain,
        domains=domains,
        shuffle=True,
        seed=args.seed,
    )
    print(f"Loaded {len(tasks)} tasks")

    evaluator = AFlowEvaluator()

    # Initialize system
    system = _build_system(args)

    # Run experiment
    all_scores = []
    domain_scores: dict[str, list[float]] = {}
    results_per_task = []
    start_time = time.time()

    for i, task in enumerate(tasks):
        try:
            result = system.process_task(task, evaluator)
            score = result["team_score"]
            all_scores.append(score)

            domain = task.get("domain", "unknown")
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(score)

            task_record: dict = {
                "task_index": i,
                "task_id": task.get("id"),
                "task_type": task.get("type"),
                "domain": domain,
                "score": score,
                "lifecycle_events": result.get("lifecycle_events", []),
            }
            # Store final_answer for code tasks to aid debugging
            if domain in ("humaneval", "mbpp") and "final_answer" in result:
                task_record["final_answer"] = result["final_answer"][:600] if result["final_answer"] else ""
            results_per_task.append(task_record)

            if (i + 1) % 10 == 0:
                recent_mean = sum(all_scores[-10:]) / 10
                elapsed = time.time() - start_time
                print(f"  Task {i+1}/{len(tasks)} | Recent mean: {recent_mean:.3f} | Elapsed: {elapsed:.0f}s")

        except KeyboardInterrupt:
            print(f"\nInterrupted at task {i}. Saving partial results...")
            break
        except Exception as e:
            print(f"  Task {i} failed: {e}")
            traceback.print_exc()
            all_scores.append(0.0)

    elapsed = time.time() - start_time

    # Compute summary metrics
    summary = summarize_results(
        system_name=args.condition,
        scores=all_scores,
        domain_scores=domain_scores,
    )
    summary["elapsed_seconds"] = elapsed
    summary["backbone_llm"] = args.backbone_llm
    summary["pool_size"] = args.pool_size
    summary["team_size"] = args.team_size
    summary["seed"] = args.seed
    summary["benchmark"] = args.benchmark

    # Save results
    output = {
        "summary": summary,
        "per_task_results": results_per_task,
        "all_scores": all_scores,
        "domain_scores": domain_scores,
        "metrics_log": system.metrics_log if hasattr(system, "metrics_log") else [],
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Mean score: {summary['mean_score']:.3f}")
    print(f"Final score (last 10): {summary['final_score']:.3f}")
    print(f"AUC: {summary['auc']:.3f}")
    print(f"Learning slope: {summary['learning_slope']:.4f}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Results saved to: {output_file}")


def _build_system(args):
    """Build the system (EvoPool variant or baseline) based on condition string."""
    condition = args.condition

    if condition == "no_memory":
        from evopool.baselines.dylan import DyLANPool
        # DyLAN with random selection = no-memory baseline
        return DyLANPool(
            pool_size=args.pool_size,
            team_size=args.team_size,
            backbone_llm=args.backbone_llm,
            seed=args.seed,
        )

    if condition == "dylan":
        from evopool.baselines.dylan import DyLANPool
        return DyLANPool(
            pool_size=args.pool_size,
            team_size=args.team_size,
            backbone_llm=args.backbone_llm,
            seed=args.seed,
        )

    if condition == "agentnet":
        from evopool.baselines.agentnet import AgentNetPool
        return AgentNetPool(
            pool_size=args.pool_size,
            team_size=args.team_size,
            backbone_llm=args.backbone_llm,
            seed=args.seed,
        )

    if condition == "self_consistency":
        from evopool.baselines.self_consistency import SelfConsistencyPool
        return SelfConsistencyPool(
            k=5,
            backbone_llm=args.backbone_llm,
            seed=args.seed,
        )

    if condition == "single_agent":
        from evopool.baselines.single_agent import SingleAgentPool
        return SingleAgentPool(
            backbone_llm=args.backbone_llm,
            seed=args.seed,
        )

    if condition == "aflow":
        from evopool.baselines.aflow import AFlowPool
        return AFlowPool(
            pool_size=args.pool_size,
            team_size=args.team_size,
            backbone_llm=args.backbone_llm,
            seed=args.seed,
        )

    # EvoPool variants
    from evopool.pool import EvoPool, PoolConfig

    codream_mode = "asymmetric"
    lifecycle_enabled = True
    collab_score_enabled = True

    if condition == "evopool_no_codream":
        codream_mode = "none"
    elif condition == "evopool_symmetric_codream":
        codream_mode = "symmetric"
    elif condition == "evopool_no_lifecycle":
        lifecycle_enabled = False
    elif condition == "evopool_no_collab_score":
        collab_score_enabled = False
    elif condition == "evopool_full":
        pass  # defaults
    elif condition == "evopool_cod_only":
        lifecycle_enabled = False
        collab_score_enabled = False
    else:
        raise ValueError(f"Unknown condition: {condition}")

    config = PoolConfig(
        pool_size_init=args.pool_size,
        team_size=args.team_size,
        codream_mode=codream_mode,
        backbone_llm=args.backbone_llm,
        lifecycle_enabled=lifecycle_enabled,
        collab_score_enabled=collab_score_enabled,
        seed=args.seed,
    )
    return EvoPool(config)


if __name__ == "__main__":
    main()
