"""
Run EvoPool on ARC-AGI-3 environments.

Unlike standard benchmarks (one-shot answer), ARC-AGI-3 requires multi-round
interaction. The pipeline per environment:

1. Team selection (from pool)
2. Warmup plays: each agent plays the environment independently to explore
3. CoDream: agents share discovered rules/strategies
4. Evaluated play: agents play again with accumulated knowledge
5. Score = best levels_completed / total_levels across team

CoDream happens between plays, not during. This matches the paper's design:
"agents debrief after a project, not during it."
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from evopool.benchmarks.arc_agi3 import (
    load_arc_agi3_tasks, ArcAGI3Evaluator, agent_play, grid_to_text
)
from evopool.llm import llm_call
from evopool.agent import Agent, AgentProfile, Experience
from evopool.selector import select_team, CollabScoreTable


def main():
    parser = argparse.ArgumentParser(description="Run EvoPool on ARC-AGI-3")
    parser.add_argument("--pool_size", type=int, default=10)
    parser.add_argument("--team_size", type=int, default=3)
    parser.add_argument("--backbone_llm", type=str, default="qwen3-8b")
    parser.add_argument("--max_steps", type=int, default=150,
                        help="Max steps per play session")
    parser.add_argument("--warmup_plays", type=int, default=2,
                        help="Warmup plays before evaluated play")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated game IDs (default: all 25)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/arc3/")
    parser.add_argument("--no_codream", action="store_true")
    parser.add_argument("--single_agent", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load environments
    game_ids = args.games.split(",") if args.games else None
    tasks = load_arc_agi3_tasks(game_ids)
    print(f"Loaded {len(tasks)} ARC-AGI-3 environments")

    # Initialize agent pool
    pool = _init_pool(args.pool_size, args.seed)
    collab_table = CollabScoreTable()
    evaluator = ArcAGI3Evaluator(max_steps_per_play=args.max_steps)

    all_results = []
    start_time = time.time()

    for task_idx, task in enumerate(tasks):
        game_id = task["game_id"]
        print(f"\n{'='*60}")
        print(f"Environment {task_idx+1}/{len(tasks)}: {task['title']} ({game_id})")
        print(f"{'='*60}")

        try:
            result = _process_environment(
                task=task,
                pool=pool,
                collab_table=collab_table,
                evaluator=evaluator,
                backbone_llm=args.backbone_llm,
                team_size=args.team_size if not args.single_agent else 1,
                warmup_plays=args.warmup_plays,
                max_steps=args.max_steps,
                no_codream=args.no_codream or args.single_agent,
            )
            all_results.append(result)

            elapsed = time.time() - start_time
            scores_so_far = [r["team_score"] for r in all_results]
            mean_score = sum(scores_so_far) / len(scores_so_far)
            print(f"  Score: {result['team_score']:.2f} | "
                  f"Running mean: {mean_score:.3f} | Elapsed: {elapsed:.0f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "game_id": game_id, "team_score": 0.0, "error": str(e)
            })

    # Save results
    elapsed = time.time() - start_time
    summary = {
        "mean_score": sum(r["team_score"] for r in all_results) / max(len(all_results), 1),
        "total_environments": len(all_results),
        "total_levels_completed": sum(r.get("levels_completed", 0) for r in all_results),
        "elapsed_seconds": elapsed,
    }
    output = {
        "summary": summary,
        "per_task_results": all_results,
        "config": vars(args),
    }
    cond = "single_agent" if args.single_agent else ("evopool_nocd" if args.no_codream else "evopool_full")
    outfile = os.path.join(args.output_dir, f"{cond}_arc_agi3_seed{args.seed}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")
    print(f"Mean score: {summary['mean_score']:.3f}")


def _init_pool(pool_size: int, seed: int) -> list[Agent]:
    """Initialize a pool of agents for ARC-AGI-3."""
    import random
    rng = random.Random(seed)
    agents = []
    for i in range(pool_size):
        profile = AgentProfile(
            persona=f"Agent-{i}: Explorer",
            skill_memory={"interactive_reasoning": 0.3 + rng.random() * 0.2},
            task_history=[],
            collab_log={},
            perf_stats={},
        )
        agents.append(Agent(profile=profile))
    return agents


def _process_environment(
    task: dict,
    pool: list[Agent],
    collab_table: CollabScoreTable,
    evaluator: ArcAGI3Evaluator,
    backbone_llm: str,
    team_size: int,
    warmup_plays: int,
    max_steps: int,
    no_codream: bool,
) -> dict:
    """
    EvoPool pipeline for one ARC-AGI-3 environment:

    For N rounds:
      1. ALL agents in pool explore independently
      2. CoDream: everyone shares discoveries → update shared knowledge
      3. Check if any agent solved it

    Score = max(levels_completed) across all agents, all rounds / total_levels
    Success = any agent completes all levels at any point.
    """
    game_id = task["game_id"]
    n_rounds = warmup_plays  # reuse this param as number of exploration rounds
    shared_knowledge = ""  # grows each round via CoDream
    best_score = 0.0
    best_levels = 0
    total_levels = None
    round_scores = []  # track progress per round

    print(f"  Pool size: {len(pool)} agents, {n_rounds} rounds")

    for rnd in range(n_rounds):
        print(f"\n  --- Round {rnd+1}/{n_rounds} ---")
        round_results = {}

        # 1. ALL agents explore independently
        for agent in pool:
            env = evaluator.make_env(game_id)

            # Build context: shared knowledge + own experience
            context = ""
            if shared_knowledge:
                context += f"[Shared knowledge from team]:\n{shared_knowledge}\n\n"
            own_exp = _build_experience_context(agent, task)
            if own_exp:
                context += own_exp

            result = agent_play(
                env=env,
                llm_call_fn=llm_call,
                model=backbone_llm,
                max_steps=max_steps,
                experience_context=context,
                agent_id=agent.agent_id,
            )
            round_results[agent.agent_id] = result

            if total_levels is None:
                total_levels = result["total_levels"]

            # Store experience
            agent.profile.experience_buffer.append(Experience(
                task_id=task["id"],
                domain="arc_agi3",
                task_type="interactive_reasoning",
                score=result["score"],
                strategy_summary=result["trajectory_summary"][:150],
                lesson=f"R{rnd+1}: {result['levels_completed']}/{result['total_levels']} in {result['actions_taken']} steps",
                source="self",
            ))

            # Track best
            if result["score"] > best_score:
                best_score = result["score"]
                best_levels = result["levels_completed"]

            if result["levels_completed"] > 0:
                print(f"    {agent.agent_id[:6]}: {result['levels_completed']}/{result['total_levels']} levels ✓")

        # Summary for this round
        scores_this_round = [r["score"] for r in round_results.values()]
        max_this_round = max(scores_this_round)
        mean_this_round = sum(scores_this_round) / len(scores_this_round)
        n_solved = sum(1 for s in scores_this_round if s > 0)
        round_scores.append({"max": max_this_round, "mean": mean_this_round, "n_solved": n_solved})
        print(f"  Round {rnd+1}: max={max_this_round:.2f}, mean={mean_this_round:.3f}, "
              f"{n_solved}/{len(pool)} agents scored >0")

        # 2. CoDream: compile all discoveries into shared knowledge
        if not no_codream:
            shared_knowledge = _compile_shared_knowledge(
                pool, round_results, task, backbone_llm,
                previous_knowledge=shared_knowledge,
            )
            print(f"  Shared knowledge updated ({len(shared_knowledge)} chars)")

        # Early exit: all levels completed
        if best_score >= 1.0:
            print(f"  SOLVED! Agent completed all levels in round {rnd+1}")
            break

        # Early exit: stagnated for 5 rounds (no improvement in max score)
        if len(round_scores) >= 5:
            recent_max = [rs["max"] for rs in round_scores[-5:]]
            if all(m == recent_max[0] for m in recent_max):
                print(f"  Stagnated for 5 rounds at max={recent_max[0]:.2f}, moving on")
                break

    # Update skill memory
    for agent in pool:
        old = agent.profile.skill_memory.get("interactive_reasoning", 0.3)
        best_own = max(
            (e.score for e in agent.profile.experience_buffer
             if e.task_id == task["id"]),
            default=0.0,
        )
        agent.profile.skill_memory["interactive_reasoning"] = 0.7 * old + 0.3 * best_own

    return {
        "game_id": game_id,
        "title": task["title"],
        "team_score": best_score,
        "levels_completed": best_levels,
        "total_levels": total_levels or 0,
        "round_scores": round_scores,
        "pool_size": len(pool),
        "n_rounds": n_rounds,
        "shared_knowledge_final": shared_knowledge[:200],
    }


def _build_experience_context(agent: Agent, task: dict) -> str:
    """Build experience context from agent's own plays of this game."""
    relevant = [
        e for e in agent.profile.experience_buffer
        if e.domain == "arc_agi3" and e.task_id == task["id"]
    ]
    if not relevant:
        return ""
    lines = ["[Your past experience with this game]:"]
    for e in relevant[-3:]:
        mark = "✓" if e.score > 0 else "✗"
        lines.append(f"- {mark} {e.strategy_summary}")
    return "\n".join(lines) + "\n\n"


def _compile_shared_knowledge(
    pool: list[Agent],
    round_results: dict[str, dict],
    task: dict,
    backbone_llm: str,
    previous_knowledge: str = "",
) -> str:
    """
    CoDream for ARC-AGI-3: merge all agents' discoveries into shared knowledge.

    Key: shared knowledge contains FACTS (rules, objects) and MULTIPLE candidate
    strategies — NOT one "best" strategy. Each agent picks which strategy to try
    based on their own experience, preserving diversity.
    """
    # Gather this round's discoveries from all agents
    discoveries = []
    for agent in pool:
        r = round_results.get(agent.agent_id)
        if not r:
            continue
        summary = r.get("trajectory_summary", "")
        levels = r["levels_completed"]
        if summary:
            discoveries.append(
                f"Agent {agent.agent_id[:6]} ({levels} levels): {summary}"
            )

    if not discoveries and not previous_knowledge:
        return ""

    discoveries_text = "\n".join(discoveries) if discoveries else "(no new discoveries)"

    prompt = (
        f"A pool of {len(pool)} agents explored an abstract puzzle game.\n\n"
    )
    if previous_knowledge:
        prompt += f"Previous shared knowledge:\n{previous_knowledge}\n\n"
    prompt += (
        f"New discoveries this round:\n{discoveries_text}\n\n"
        "Update the shared knowledge. Include:\n"
        "1. CONFIRMED RULES: what the actions do, what objects mean (facts everyone agrees on)\n"
        "2. CANDIDATE STRATEGIES: list ALL different approaches agents tried, "
        "noting which succeeded and which failed. Do NOT pick one best strategy — "
        "agents should choose which to try based on their own experience.\n"
        "3. OPEN QUESTIONS: things still unknown that need more exploration\n\n"
        "Be concise. This knowledge will be shared with all agents for the next round."
    )

    try:
        return llm_call(model=backbone_llm, user=prompt, max_tokens=500)
    except Exception:
        return previous_knowledge + "\n" + discoveries_text[:300]


if __name__ == "__main__":
    main()
