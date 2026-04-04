# Research Brief: EvoPool — Evolving Agent Pool with Co-Dream

## Problem Statement

Current multi-agent LLM systems treat agent pools as **static artifacts**: a fixed set of agents with predefined roles is assembled once and never changes, regardless of how many tasks the system has processed. When a new task arrives, the system selects from the same frozen pool it started with. This design has two fundamental failures:

1. **Team Amnesia**: Knowledge accumulated during collaboration is discarded after each task. Agents that worked together on Task 1 bring no carry-over learning to Task 2, even if the tasks are related.

2. **Static Specialization**: Agent roles are manually pre-designed (coder, tester, planner) rather than emerging organically from task experience. The pool never discovers that it needs a "debugging specialist" or that two agents have become functionally redundant.

There is no existing work that simultaneously addresses (a) a **dynamically evolving pool** where agent profiles change over time, (b) **co-dream** — a mechanism for agents that collaborated on a task to jointly consolidate cross-pollinated learnings, and (c) **pool lifecycle operators** (specialize, fork, merge, prune, genesis) that reshape pool composition over a lifelong task stream.

## Background

- **Field**: LLM-based multi-agent systems, lifelong learning
- **Sub-area**: Agent memory, auto multi-agent system design, continual learning
- **Target venue**: NeurIPS 2026

**Key papers already surveyed:**
- ADAS (ICLR'25): meta agent searches agentic designs, has archive but no cross-task memory
- AFlow (ICLR'25 Oral): MCTS workflow search, experience only within one run
- Evolving Orchestration (NeurIPS'25): fixed pool, evolving selection policy
- DyLAN: dynamic selection from static pool, Agent Importance Score
- MAE (Multi-Agent Evolve): 3 fixed roles co-evolve via RL, role structure hardcoded
- LIET (2506.07232): shared cooperation list in embodied settings, no persistent pool evolution
- Darwin Gödel Machine (2505.22954): single-agent code rewriting + archive, not multi-agent specialist pool
- A-MEM (NeurIPS'25): best single-agent memory baseline (LoCoMo, DialSim)
- AWM (ICML'25): single-agent workflow memory, +24.6% Mind2Web, +51.1% WebArena
- SWE-Bench-CL (2507.00014): continual coding benchmark with forward/backward transfer metrics
- EvoMem (2511.20857, DeepMind): single-agent self-evolving memory benchmark
- EvoMem (2511.01912): multi-agent planning with dual memory, but resets after each query

**The key gap (2x2 matrix):**
```
                 Pool Static       Pool Evolving
Agent Static   | DyLAN, ADAS     |  (empty)
Agent Evolving | MAE (3 roles)   |  EvoPool ← our work
```

## Core Method: EvoPool

### Three components:

**1. Individual Evolution**
Each agent maintains a structured profile: `{persona, skill_memory, task_history, performance_stats}`. After each task, the agent updates its profile based on feedback. Specialization emerges naturally — no manual role assignment.

**2. Co-Dream (key contribution)**
After agents A, B, C complete a task together, they run an offline joint consolidation pass:
- A extracts what it learned from working with B and C
- B updates its profile with patterns observed from A and C
- C integrates complementary knowledge from A and B
- Each agent's private profile is updated; no centralized shared memory
This is inspired by Claude Code's Auto-Dream mechanism (single-agent memory consolidation), extended to multi-agent cross-pollination.

**3. Pool Lifecycle Operators**
- **Specialize**: agent consistently strong in domain X → reinforce that direction
- **Fork**: agent handling two divergent task types → spawn two specialized variants
- **Merge**: two agents with high profile similarity → consolidate into one richer agent
- **Prune**: agent consistently underperforms → remove from pool
- **Genesis**: pool lacks coverage for emerging task type → spawn new agent from best generalist

**Selection Policy**:
`score(agent_i, task) = affinity(profile_i, task_type) + diversity_bonus + historical_collab_score(team)`

`historical_collab_score` is learned from co-dream history — some agent pairs develop persistent "synergy."

## Constraints

- **Compute**: OSU HPC cluster, hw-grp partition (H100 GPUs available via Slurm)
- **Slurm account**: hw-grp
- **GPU request**: `srun -A hw-grp -p hw-grp --gres=gpu:1 --pty bash`
- **Timeline**: NeurIPS 2026 submission (May 2026)
- **Target venue**: NeurIPS 2026

## What I'm Looking For

- [x] New method: EvoPool — evolving agent pool with co-dream and lifecycle operators
- [x] Improvement on existing methods: beat DyLAN (static pool), ADAS, Evolving Orchestration (NeurIPS'25), A-MEM
- [ ] Not a survey/benchmark paper — we use existing benchmarks

## Benchmarks to Target

1. **SWE-Bench-CL** — sequential coding tasks, natural for specialization emergence
2. **AFlow's benchmarks (HotpotQA, MBPP, MATH, HumanEval, GSM8K, DROP)** — as sequential streams
3. **LoCoMo + multi-agent extension** — beat A-MEM (NeurIPS'25) baseline
4. **GAIA** — diverse task types, ideal for measuring specialization diversity

## Baselines to Beat

1. DyLAN (static pool + dynamic selection) — strongest static pool baseline
2. Evolving Orchestration (NeurIPS'25) — strongest evolving orchestration baseline
3. A-MEM multi-agent extension — strongest memory baseline
4. No-memory ADAS/AFlow — vanilla auto-MAS

## Key Metric

- **Lifelong performance curve**: task performance vs. number of tasks processed (should keep rising)
- **Specialization emergence**: diversity of agent profiles over time (entropy of skill distributions)
- **Selection efficiency**: accuracy of task-agent matching improves over time
- **Co-dream ablation**: +/- co-dream vs. individual-only memory update

## Domain Knowledge & Intuitions

- The co-dream mechanism is the hardest part to get right — the challenge is preventing homogenization (if all agents learn from each other equally, they converge to the same profile)
- Solution: co-dream should be **asymmetric** (A learns from B in B's areas of strength, not everything)
- Pool size N should be large (50-100), but effective team size k small (3-5); this ratio drives specialization pressure
- The fork/merge/prune operators should be rare events (triggered by sustained patterns, not single-task signals) to avoid instability

## Non-Goals

- Not doing fine-tuning / weight updates — prompt-based evolution only
- Not building a new benchmark — use existing ones
- Not doing MARL / reward shaping — LLM-based
- Not reproducing Claude Code's exact Dream implementation — inspired by it conceptually

## Existing Results

None yet — this is the starting point. The research brief captures prior literature analysis done via web search.
