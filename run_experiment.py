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
  memcollab               MemCollab (arXiv 2603.23234): contrastive trajectory distillation → shared memory
  evomem                  EvoMem: individual evolving memory (Reflexion-style self-reflection, no cross-agent sharing)

Available benchmarks:
  aflow_stream     6-domain sequential stream (GSM8K/HotpotQA/MBPP/MATH/HumanEval/DROP)
  gsm8k_stream     GSM8K only (for pilot experiments)
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback

from evopool.benchmarks.aflow_stream import load_aflow_stream, AFlowEvaluator
from evopool.benchmarks.hard_math_stream import load_hard_math_stream, HardMathEvaluator
from evopool.benchmarks.hard_code_stream import load_hard_code_stream, HardCodeEvaluator
from evopool.eval.metrics import summarize_results, print_comparison_table


def main():
    parser = argparse.ArgumentParser(description="Run EvoPool experiment")
    parser.add_argument("--condition", type=str, required=True, help="System to evaluate")
    parser.add_argument("--benchmark", type=str, default="aflow_stream")
    parser.add_argument("--n_tasks", type=int, default=60, help="Total tasks in stream")
    parser.add_argument("--n_per_domain", type=str, default="10",
                        help="Tasks per domain (int or 'all' for all available)")
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--team_size", type=int, default=3)
    parser.add_argument("--backbone_llm", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domain list (default: all 6)")
    parser.add_argument("--shuffle_all", action="store_true",
                        help="Shuffle all tasks across domains (destroys domain ordering)")
    parser.add_argument("--save_pool", type=str, default=None,
                        help="Path to save final pool state (EvoPool only)")
    parser.add_argument("--load_pool", type=str, default=None,
                        help="Path to load pre-trained pool state (EvoPool only, warm start)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{args.condition}_{args.benchmark}_seed{args.seed}.json"
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
        n_per_domain = None if args.n_per_domain == "all" else int(args.n_per_domain)

    print(f"Loading tasks (benchmark={args.benchmark}, n_per_domain={n_per_domain})...")
    if args.benchmark == "hard_math_stream":
        tasks = load_hard_math_stream(
            domains=domains,
            n_per_domain=n_per_domain,
            seed=args.seed,
            shuffle=True,
        )
        evaluator = HardMathEvaluator()
    elif args.benchmark == "hard_code_stream":
        tasks = load_hard_code_stream(
            domains=domains,
            n_per_domain=n_per_domain,
            seed=args.seed,
            shuffle=True,
        )
        evaluator = HardCodeEvaluator()
    else:
        tasks = load_aflow_stream(
            n_per_domain=n_per_domain,
            domains=domains,
            shuffle=True,
            seed=args.seed,
        )
        evaluator = AFlowEvaluator()
    print(f"Loaded {len(tasks)} tasks")

    # Optionally shuffle all tasks across domains (E56 ablation)
    if args.shuffle_all:
        import random as _rng
        _rng.Random(args.seed).shuffle(tasks)
        print(f"  Shuffled all {len(tasks)} tasks across domains")

    # Initialize system
    system = _build_system(args)

    # Run experiment (with checkpoint for crash recovery)
    all_scores = []
    domain_scores: dict[str, list[float]] = {}
    results_per_task = []
    start_time = time.time()
    checkpoint_file = os.path.join(output_dir, "_checkpoint.json")

    # Resume from checkpoint if exists
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            ckpt = json.load(open(checkpoint_file))
            results_per_task = ckpt.get("per_task_results", [])
            start_idx = len(results_per_task)
            all_scores = [r["score"] for r in results_per_task]
            for r in results_per_task:
                d = r.get("domain", "unknown")
                if d not in domain_scores:
                    domain_scores[d] = []
                domain_scores[d].append(r["score"])
            print(f"Resumed from checkpoint: {start_idx} tasks done")
        except Exception:
            start_idx = 0

    # Heartbeat: print every 2 min so watchdog doesn't kill long tasks
    _heartbeat_stop = threading.Event()
    def _heartbeat():
        while not _heartbeat_stop.wait(120):
            print(f"  [heartbeat] alive, processing task... ({time.strftime('%H:%M:%S')})", flush=True)
    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    for i, task in enumerate(tasks):
        if i < start_idx:
            continue  # skip already-completed tasks
        try:
            print(f"  Starting task {i+1}/{len(tasks)} ({task.get('domain','?')}/{task.get('id','?')})...", flush=True)
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
                "codream_generated": result.get("codream_generated", 0),
                "codream_verified": result.get("codream_verified", 0),
                "codream_verify_rate": result.get("codream_verify_rate", 0.0),
                "codream_insight_texts": result.get("codream_insight_texts", []),
            }
            # Store final_answer for code tasks to aid debugging
            if domain in ("humaneval", "mbpp") and "final_answer" in result:
                task_record["final_answer"] = result["final_answer"][:600] if result["final_answer"] else ""
            results_per_task.append(task_record)

            if (i + 1) % 10 == 0:
                recent_mean = sum(all_scores[-10:]) / 10
                elapsed = time.time() - start_time
                print(f"  Task {i+1}/{len(tasks)} | Recent mean: {recent_mean:.3f} | Elapsed: {elapsed:.0f}s")
                # Checkpoint every 10 tasks for crash recovery
                json.dump({"per_task_results": results_per_task}, open(checkpoint_file, "w"))

        except KeyboardInterrupt:
            print(f"\nInterrupted at task {i}. Saving partial results...")
            json.dump({"per_task_results": results_per_task}, open(checkpoint_file, "w"))
            break
        except Exception as e:
            err_str = str(e)
            if "Connection" in err_str or "refused" in err_str:
                # vLLM likely restarting — wait and retry this task
                print(f"  Task {i} connection error, waiting 60s for vLLM recovery...")
                time.sleep(60)
                try:
                    result = system.process_task(task, evaluator)
                    score = result["team_score"]
                    all_scores.append(score)
                    domain = task.get("domain", "unknown")
                    if domain not in domain_scores:
                        domain_scores[domain] = []
                    domain_scores[domain].append(score)
                    results_per_task.append({
                        "task_index": i, "task_id": task.get("id"),
                        "task_type": task.get("type"), "domain": domain,
                        "score": score, "retried": True,
                    })
                    continue
                except Exception as e2:
                    print(f"  Task {i} retry also failed: {e2}")
            print(f"  Task {i} failed: {e}")
            traceback.print_exc()
            all_scores.append(0.0)

    _heartbeat_stop.set()
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

    # Save pool state if requested (EvoPool only)
    if args.save_pool and hasattr(system, "save"):
        system.save(args.save_pool)
        print(f"Pool state saved to: {args.save_pool}")

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

    if condition == "memcollab":
        from evopool.baselines.memcollab import MemCollabPool
        return MemCollabPool(
            pool_size=args.pool_size,
            team_size=args.team_size,
            backbone_llm=args.backbone_llm,
            seed=args.seed,
        )

    if condition == "evomem":
        from evopool.baselines.evomem import EvoMemPool
        return EvoMemPool(
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

    disable_l3 = False
    disable_l2 = False
    team_random = False

    if condition == "evopool_no_codream":
        codream_mode = "none"
    elif condition == "evopool_symmetric_codream":
        codream_mode = "symmetric"
    elif condition == "evopool_no_lifecycle":
        lifecycle_enabled = False
    elif condition == "evopool_no_collab_score":
        collab_score_enabled = False
    elif condition == "evopool_no_l3":
        disable_l3 = True    # E22: CoDream L1+L2 only, no cross-domain broadcast
    elif condition == "evopool_no_l2":
        disable_l2 = True    # E23: CoDream L1+L3 only, no subdomain accumulation
    elif condition == "evopool_random_team":
        team_random = True   # E24: random team selection
    elif condition == "evopool_full":
        pass  # defaults
    elif condition == "evopool_enhanced_codream":
        pass  # E25: enhanced CoDream (set below via codream_enhanced=True)
    elif condition == "evopool_no_verify":
        pass  # E27: skip verify gate
    elif condition == "evopool_cod_only":
        lifecycle_enabled = False
        collab_score_enabled = False
    else:
        raise ValueError(f"Unknown condition: {condition}")

    codream_enhanced = (condition == "evopool_enhanced_codream")
    codream_no_verify = (condition == "evopool_no_verify")

    config = PoolConfig(
        pool_size_init=args.pool_size,
        team_size=args.team_size,
        codream_mode=codream_mode,
        backbone_llm=args.backbone_llm,
        lifecycle_enabled=lifecycle_enabled,
        collab_score_enabled=collab_score_enabled,
        codream_disable_l3=disable_l3,
        codream_disable_l2=disable_l2,
        team_selection_random=team_random,
        codream_enhanced=codream_enhanced,
        codream_no_verify=codream_no_verify,
        seed=args.seed,
    )

    # Warm-start: load pre-evolved pool state, override config's fresh pool
    if getattr(args, "load_pool", None):
        pool = EvoPool.load(args.load_pool)
        # Override config to match current run parameters (keeps memories, resets task_index)
        pool.config = config
        pool.task_index = 0
        print(f"Warm-start: loaded pool from {args.load_pool} "
              f"({len(pool.pool)} agents, memories intact)")
        return pool

    return EvoPool(config)


if __name__ == "__main__":
    main()
