# Idea Discovery Report

**Direction**: EvoPool — Evolving Agent Pool with Co-Dream (NeurIPS 2026)
**Generated**: 2026-04-03
**Pipeline**: RESEARCH_BRIEF.md → novelty-check → research-review → research-refine-pipeline
**Ideas evaluated**: 1 (pre-formed in RESEARCH_BRIEF.md) → validated → RECOMMENDED

---

## Executive Summary

EvoPool proposes a lifelong multi-agent system where a pool of N~100 LLM agents evolves across a task stream via three mechanisms: per-agent profile evolution, **Co-Dream** (asymmetric cross-agent memory consolidation after collaboration), and **Pool Lifecycle Operators** (fork/merge/prune/genesis/specialize). The novelty check confirms overall 7/10 novelty with the Co-Dream mechanism having zero direct literature overlap and high novelty. The closest threat is AgentNet (NeurIPS 2025), which must be clearly distinguished. Recommend proceeding with Co-Dream + Pool Lifecycle as the twin primary contributions, repositioning per-agent profiles as an enabling substrate.

---

## Literature Landscape

### Current State (as of 2026-04) — Updated after full survey

**Static pool systems** (DyLAN, AgentPrune, AgentDropout): Select from a fixed set of agents per-task. No memory, no evolution. DyLAN's Agent Importance Score is computed within each task and discarded.

**Static design search** (ADAS, AFlow, EvoAgentX): Search for good agent configurations offline. The discovered system is then frozen. No cross-task learning.

**Evolving orchestration** (NeurIPS'25, arXiv:2505.19591): RL policy learns which agents to sequence, but agent identities are static. The "evolution" is in the selection policy, not the agents themselves.

**EvoAgentX** (EMNLP'25 Demo, arXiv:2507.03616): Automates generation and optimization of multi-agent workflows via TextGrad/AFlow/MIPRO. Evolves workflow topology and prompts — but agents have no persistent profiles or cross-agent memory. A workflow optimizer, not a pool manager.

**CoMAS** (2510.08529): Multi-agent co-evolution via RL interaction rewards (REINFORCE++). Fixed 3-role structure (Solver/Evaluator), weight updates (not memory/profile), no pool lifecycle. Must differentiate: EvoPool is prompt-only, scales to N=100, and does asymmetric post-collaboration consolidation.

**MemCollab** (2603.23234, 2026): **NEW HIGH THREAT**. Cross-agent memory collaboration via contrastive trajectory distillation — builds agent-agnostic shared memory by contrasting trajectories from different agents. Key differentiator: MemCollab aims for homogenization (shared neutral memory); EvoPool Co-Dream is asymmetric + strength-directed (preserves specialization, prevents convergence).

**AgentSpawn** (2602.07072, 2026): Dynamic spawning of child agents within a single task. Covers Fork/Genesis dimension but is single-task scoped — no persistent pool, no Merge/Prune, no cross-task memory.

**Per-agent persistent memory** (AgentNet NeurIPS'25, arXiv:2504.00587): **MOST IMPORTANT COMPETITOR**. Each agent has RAG-based memory of task trajectories; topology adapts dynamically. Agents specialize through retrieval. Does NOT do: cross-agent asymmetric knowledge transfer, explicit pool lifecycle operators (fork/merge/genesis), pairwise synergy tracking.

**ASpec** (NeurIPS'25 Workshop): Discovers specialist archetypes via evolutionary search; retain-then-escalate restructuring policy. Small team setting (not N~100 pool), no fork/merge/genesis semantics, no cross-agent memory transfer.

**Single-agent memory** (A-MEM NeurIPS'25, AWM ICML'25, EvoMem DeepMind): Strong single-agent baselines but no multi-agent context.

**MAE (Multi-Agent Evolve)**: 3 hardcoded roles co-evolve via RL — the role structure is fixed, no profile emergence.

### 2×2 Gap Matrix

```
                  Pool Static          Pool Evolving
Agent Static   | DyLAN, AgentPrune   | Evolving Orchestration
Agent Evolving | AgentNet (partial)  | EvoPool ← our work
                 ASpec (partial)
```

AgentNet occupies the bottom-left with partial coverage but lacks the pool-level lifecycle algebra and cross-agent transfer. The true bottom-right quadrant remains essentially empty.

---

## Recommended Ideas (ranked)

### 🏆 Idea 1: EvoPool — Evolving Agent Pool with Co-Dream

**Hypothesis**: A pool of LLM agents can develop persistent specialization across a long task stream if (a) agents maintain structured profiles that evolve with experience, (b) collaborating agents asymmetrically cross-pollinate knowledge after each task (Co-Dream), and (c) the pool composition is managed through principled lifecycle operators. This will produce a rising lifelong performance curve and measurably distinct agent specializations, unlike any static-pool baseline.

**Core Method**:
1. **Individual Evolution**: Profile `{persona, skill_memory, task_history, performance_stats}` updated post-task from feedback
2. **Co-Dream** (PRIMARY): After A,B,C collaborate, each privately updates profile in directions where collaborators are stronger (asymmetric to prevent homogenization)
3. **Pool Lifecycle**: Specialize / Fork / Merge / Prune / Genesis triggered by sustained profile patterns

**Selection Policy**: `score(agent_i, task) = affinity(profile_i, task_type) + diversity_bonus + historical_collab_score(team)`

**Novelty**: 7/10 overall
- Co-Dream: HIGH (zero direct overlap)
- Pool Lifecycle operators: HIGH (closest is ASpec workshop paper)
- Per-agent profiles: MEDIUM-LOW (AgentNet does this; frame as substrate)
- Historical collab score: MEDIUM-HIGH (DyLAN is transient only)

**Feasibility**:
- Compute: ~100-200 GPU-hours on H100s (OSU hw-grp)
- Data: SWE-Bench-CL (available), GAIA (available), AFlow benchmarks (available)
- Implementation: ~2-3 weeks; prompt-based only, no fine-tuning
- Backbone LLM: Claude Sonnet 4.6 or GPT-4o for agents

**Risk**: MEDIUM — main risk is that Co-Dream shows weak signal vs. individual-only update. Mitigation: run ablation on AFlow stream first (cheap) to get pilot signal.

**Contribution type**: New method + empirical evaluation

**Reviewer's likely objections**:
1. "How is this different from AgentNet?" → Key: AgentNet has per-agent RAG retrieval; EvoPool has cross-agent asymmetric knowledge transfer + pool lifecycle algebra
2. "Why not just have a growing pool (add specialists as needed)?" → Show: lifecycle operators (fork/merge/prune) maintain pool at bounded size N while improving specialization diversity
3. "Co-Dream might just homogenize agents" → Directly ablate symmetric vs. asymmetric Co-Dream; show entropy of skill distributions over time

**Why proceed**: The Co-Dream mechanism is genuinely novel, has zero literature precedent, and is theoretically well-motivated (asymmetric professional learning analogy). The pool lifecycle operators provide interpretable structure that black-box RL restructuring (ASpec) lacks. Together they make a compelling story: "agents that learn not just from tasks but from each other, within a managed living pool."

**Pilot plan**: Run EvoPool (Co-Dream only, no lifecycle operators) on AFlow's GSM8K stream (30 tasks) vs. AgentNet-style per-agent memory baseline. If Co-Dream shows ≥2% improvement → POSITIVE signal. Estimated: 2-4 GPU-hours.

---

## Eliminated Ideas

| Idea | Reason eliminated |
|------|-------------------|
| Static Pool + Better Selection | DyLAN already does this; insufficient novelty |
| Agent Profile Evolution (no Co-Dream) | AgentNet already does per-agent memory; insufficient differentiation |
| Pool Lifecycle Only (no Co-Dream) | ASpec does evolutionary restructuring; weaker novelty story |

---

## Novelty Check Results

Full report: see novelty check summary above and in project context.

| Claim | Novelty | Closest Paper |
|-------|---------|--------------|
| Evolving pool composition (lifecycle ops) | MEDIUM | AgentNet, ASpec |
| Per-agent persistent profile | MEDIUM-LOW | AgentNet, A-MEM |
| **Co-Dream asymmetric consolidation** | **HIGH** (closest: MemCollab 2603.23234) | MemCollab — symmetric/agnostic; Co-Dream is strength-directed |
| **Pool lifecycle operators** | **HIGH** | ASpec (workshop) |
| Historical collab score | MEDIUM-HIGH | DyLAN (transient) |

**Key differentiators to stress in paper**:
1. Co-Dream anti-homogenization property (show empirically: symmetric Co-Dream collapses profiles; asymmetric doesn't)
2. Fork-on-divergence vs. always-expanding pool (lifecycle maintains bounded N)
3. Interpretable pool coverage analysis (genesis fires when capability gap detected)
4. Pair-synergy vs. per-agent selection (ablate historical_collab_score)

---

## Next Steps

- [x] IDEA_REPORT.md generated
- [ ] `refine-logs/FINAL_PROPOSAL.md` — refined method description
- [ ] `refine-logs/EXPERIMENT_PLAN.md` — claim-driven experiment roadmap
- [ ] `refine-logs/EXPERIMENT_TRACKER.md` — run tracker
- [ ] Implement pilot experiment (AFlow stream + Co-Dream ablation)
- [ ] Submit pilot jobs to hw-grp H100 via Slurm
- [ ] `/auto-review-loop` after results
