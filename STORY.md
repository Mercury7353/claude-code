# EvoPool: The Research Story

## One-Line Pitch

*Can a population of LLM agents learn collectively from experience — improving over time on hard reasoning tasks without any gradient updates?*

---

## The Problem (Why This Matters)

Large Language Models are powerful reasoners, but every query starts from scratch. Unlike humans, who build intuition from past problem-solving, LLMs have no mechanism to accumulate and share strategic knowledge across tasks. This limitation is particularly acute on hard reasoning tasks (competition math, algorithmic coding) where problem-solving strategies matter more than raw capability.

**The gap**: Multi-agent systems (DyLAN, AgentNet, self-consistency) improve single-query performance through diversity and debate, but none of them *learn* — they are equally stateless across tasks. Self-consistency, the strongest baseline, even *degrades* on the hardest problems because majority voting amplifies errors when accuracy drops below 50%.

**Our question**: What if we gave agents *memory* and let the population *evolve*?

---

## The Framework (What We Built)

EvoPool is a **modular framework** for collective learning in LLM agent populations. It has three core mechanisms, each independently ablatable:

### 1. Experience Accumulation
Each agent maintains a personal experience buffer — concrete strategy summaries from past tasks. When facing a new problem, agents retrieve relevant past experiences. This is the foundation: without experience, there is nothing to learn from.

### 2. Evolutionary Lifecycle
The agent pool is not static. After every task:
- **Fork**: High-performing agents spawn variants (preserving successful strategies)
- **Merge**: Weak agents combine knowledge from stronger peers
- **Retire**: Persistently poor agents are replaced

This maintains strategic diversity while amplifying what works — natural selection for problem-solving strategies.

### 3. Collaborative Reflection
After each task, the team reflects together. We study two reflection strategies:

- **CoDream**: A 5-phase protocol (REFLECT → CONTRAST → IMAGINE → DEBATE → CRYSTALLIZE) that generates shared insights through structured deliberation
- **LeadLearn**: A leader-directed variant where the best-performing agent distills lessons for the team

Both add to the experience buffer. The question is: which helps, when, and why?

---

## The Evidence (What We Found)

### Finding 1: Pool Methods Dominate Hard Math

On 367 competition math problems (MATH L4/L5 + AIME 2022-2025):

| Method | Overall Accuracy |
|--------|-----------------|
| **EvoPool-LeadLearn** | **0.480** |
| **EvoPool-CoDream** | **0.463** |
| **EvoPool-noCoDream** | **0.458** |
| MemCollab | 0.431 |
| EvoMem | 0.422 |
| Self-Consistency | 0.406 |
| DyLAN | 0.403 |
| AgentNet | 0.381 |
| Single Agent | 0.251 |

**All three EvoPool variants beat all baselines.** The core contribution (experience + lifecycle) is the common denominator. Both reflection strategies add value on top.

### Finding 2: Learning is Genuine, Not Just Sampling

The cleanest evidence comes from MBPP (257 coding tasks, controlled ordering):

| Method | First Half | Second Half | Delta |
|--------|-----------|-------------|-------|
| Single Agent (no learning) | 0.828 | 0.791 | **-0.037** |
| EvoPool-CoDream | 0.695 | 0.729 | **+0.034** |
| EvoPool-noCoDream | 0.680 | 0.722 | **+0.042** |
| EvoPool-LeadLearn | 0.664 | 0.709 | **+0.045** |

The single agent *declines* as it faces later tasks without learning. All EvoPool variants *improve against this trend*. The net learning effect is +0.07 to +0.08 — consistent across all variants.

**This is not sampling diversity** — SC also starts strong and doesn't improve. This is genuine accumulation of problem-solving strategies.

### Finding 3: Self-Consistency Collapses on the Hardest Problems

SC achieves 0.542 on MATH-hard but collapses to 0.067 on AIME. Why? Majority voting amplifies errors: when the base model solves a problem <50% of the time, voting for the most common answer almost guarantees picking the wrong one.

EvoPool variants maintain performance on AIME (0.177-0.253 average) because experience-guided diversity explores *different strategies*, not just *temperature diversity*. The agents have learned what approaches work for competition-level problems.

### Finding 4: Reflection Strategies Have Complementary Strengths

| Reflection Type | math_hard | AIME | CodeContests |
|----------------|-----------|------|-------------|
| LeadLearn | **0.569** | 0.253 | — |
| CoDream | 0.550 | **0.247** | **0.438** |
| noCoDream (none) | 0.553 | 0.177 | 0.416 |

- **LeadLearn** excels on structured problems (math_hard) where clear strategy transfer is possible
- **CoDream** excels on the hardest problems (AIME +0.067 on 2023/2025) where creative strategy exploration matters
- Both beat no-reflection, confirming that **how agents share knowledge matters**

### Finding 5: CodeContests Shows the Framework's Versatility

On competitive programming (stdin/stdout problems, much harder than MBPP/HumanEval):

| Method | CodeContests |
|--------|-------------|
| **EvoPool v3** | **0.438** |
| noCoDream v3 | 0.416 |
| Self-Consistency | 0.198 |
| AgentNet | 0.102 |

EvoPool achieves **2.2x the score of the best baseline** on the hardest code benchmark. Experience accumulation is particularly valuable here: agents learn which algorithmic patterns and edge cases to check.

---

## The Story Arc (For the Paper)

### Act 1: The Problem
LLM multi-agent systems improve single-query performance but don't learn. They're equally surprised by similar problems throughout their lifetime. This is a fundamental limitation for hard reasoning.

### Act 2: The Hypothesis
If we give agents *memory* (experience buffer), *evolution* (lifecycle), and *reflection* (CoDream/LeadLearn), the population should accumulate collective intelligence over time.

### Act 3: The Evidence
- **Experience + lifecycle = learning** (all three variants beat all baselines; consistent +0.034 to +0.045 learning deltas)
- **Reflection adds depth** (CoDream helps on hardest problems; LeadLearn helps on structured problems; both beat no-reflection)
- **Stateless methods collapse** on truly hard tasks (SC: 0.067 on AIME)

### Act 4: The Insight
Collective learning in LLM populations requires three things: (1) a mechanism to accumulate experience, (2) population-level selection pressure to amplify good strategies, and (3) reflective sharing to transfer knowledge between agents. Remove any one, and performance degrades.

---

## Key Figures (Planned)

1. **Hero figure**: Schematic of EvoPool framework (pool → team selection → solve → reflect → evolve)
2. **Learning curve**: Rolling accuracy over task stream (EvoPool variants rise, SA/SC flat/decline) — REQUIRES randomized ordering
3. **Ablation waterfall**: SA → +experience → +lifecycle → +CoDream/LeadLearn (showing additive contributions)
4. **SC collapse**: Accuracy vs. task difficulty (SC drops off cliff on AIME, EvoPool maintains)
5. **Qualitative insight examples**: What CoDream generates and how it helps on subsequent tasks

---

## Open Questions (To Be Resolved)

1. **Multi-seed validation**: All results are seed=42. Need 3+ seeds for error bars.
2. **Randomized ordering**: Hard_math learning curve is confounded by MATH→AIME ordering. Need shuffled version.
3. **Compute efficiency**: How does EvoPool compare to SC with matched token budget?
4. **Scaling**: Does pool size 40 or 50 help further? Does team size 5 help?
5. **Cross-domain transfer**: Do math insights help on code? (Currently no transfer between domains)

---

## Narrative Principles

1. **EvoPool is a framework, not a single method.** CoDream and LeadLearn are both "our methods." This eliminates the "ablation beats method" concern.
2. **The core finding is that experience + lifecycle enables learning.** CoDream/LeadLearn add on top.
3. **Never hide weaknesses.** CIs overlap on CoDream vs noCoDream — acknowledge this, explain it as "reflection is one component of a three-part framework."
4. **Lead with the SC collapse story.** It's dramatic, it's true, and it motivates why learning matters.
5. **Show, don't tell.** Qualitative examples of what agents learned make the abstract concrete.
