# NeurIPS Mock Review — EvoPool (2026-04-08)

## Score: 4/10 (Borderline Reject) | Confidence: 4/5

---

## Summary

EvoPool proposes a framework for collective learning among a pool of LLM agents. A population of 20 agents (teams of 3) tackles reasoning tasks sequentially, accumulating experience and collaboratively reflecting via a 5-phase "CoDream" protocol. An evolutionary lifecycle (fork/merge/retire) maintains population diversity. Evaluated on hard math (MATH L4/L5 + AIME) and code (MBPP + HumanEval + CodeContests) benchmarks.

## Strengths

1. **Compelling problem formulation** — LLM agent populations learning collectively without gradient updates is timely and intellectually interesting
2. **Comprehensive baselines** — SA, SC, DyLAN, AgentNet, EvoMem, MemCollab + multiple ablation variants
3. **Genuine learning signal** — MBPP first/second half analysis shows clean positive deltas (+0.034 to +0.045) while SA declines (-0.037)
4. **Strong absolute code performance** — CodeContests 0.438 (vs SC 0.198) is a notable result
5. **Creative mechanism design** — CoDream 5-phase protocol and lifecycle management are well-motivated

## Weaknesses

### Major
- **W1**: Flagship method loses to its own ablation (LeadLearn 0.480 > EvoPool 0.463)
- **W2**: CoDream contribution not statistically significant (+0.005 on full run, CIs fully overlap)
- **W3**: Single seed (42), no error bars on primary results
- **W4**: Math learning curve confounded by difficulty ordering (Q1=0.615→Q4=0.245 = task difficulty, not learning failure)

### Moderate
- **W5**: v3 code results incomplete (450/586), projected numbers
- **W6**: SC collapse on AIME (0.067) may be voting artifact, not fair comparison
- **W7**: No computational cost analysis (20 agents + 5-phase CoDream overhead)

### Minor
- **W8**: No qualitative analysis of what CoDream actually generates

## Questions for Authors

1. Can you report multi-seed results (3+ seeds)?
2. What happens with randomized task ordering on hard_math?
3. Total inference cost (tokens) for EvoPool vs SC with matched compute?
4. What was "fixed" in the fixed-run CoDream ablation?
5. Pool size scaling behavior?

---

## TOP 3 DAMAGING WEAKNESSES

1. **CoDream empirically unjustified** — +0.005 on main benchmark, the paper's most novel contribution adds nothing measurable
2. **LeadLearn > EvoPool** — paper named after the worse variant
3. **Single seed + overlapping CIs** — no claim has statistical backing

---

## PROPOSED REFRAMING

### From: "EvoPool with CoDream is THE method"
### To: "What mechanisms enable LLM agent populations to learn collectively?"

**Key insight**: The paper's REAL finding is that **experience accumulation + evolutionary lifecycle** is the core driver. CoDream and LeadLearn are **two alternative reflection strategies** with complementary strengths. This is a richer, more defensible story.

**New title options:**
- "Collective Learning in LLM Agent Populations: An Empirical Study of Experience, Evolution, and Reflection"
- "EvoPool: How Agent Populations Learn from Experience on Hard Reasoning Tasks"

---

## MINIMUM ADDITIONAL EXPERIMENTS

### Must-Have (not submittable without)
| # | Experiment | Purpose | Cost |
|---|-----------|---------|------|
| E1 | 3 seeds on top-5 methods (EvoPool, noCoDream, LeadLearn, SC, MemCollab) on hard_math | Statistical rigor | 3x current |
| E2 | Randomized task ordering on hard_math (EvoPool + noCoDream) | Clean learning curve | 2 runs |
| E3 | Complete CodeContests (136 remaining tasks) | Remove "projected" | marginal |

### Strongly Recommended
| # | Experiment | Purpose | Cost |
|---|-----------|---------|------|
| E4 | Compute-matched SC (k=20) on hard_math | Strongest baseline | 1 run |
| E5 | Difficulty-stratified analysis | Show learning per stratum | analysis only |
| E6 | Qualitative CoDream examples | Make contribution tangible | zero compute |

### Nice-to-Have
| # | Experiment | Purpose | Cost |
|---|-----------|---------|------|
| E7 | Cost analysis (tokens per method) | Efficiency argument | logging |
| E8 | Pool size ablation (5, 10, 20, 40) | Scaling story | 3 runs |

---

## REFINED PAPER OUTLINE

### Title: "EvoPool: Collective Learning in Evolutionary Agent Pools for Hard Reasoning"

**Section 1: Introduction** (2p)
- Motivation: LLMs are stateless per query. Can agent populations accumulate knowledge?
- Key question: What mechanisms enable collective learning without weight updates?
- Contribution 1: EvoPool framework (experience + lifecycle + reflection)
- Contribution 2: Systematic ablation identifying which mechanisms drive learning
- Contribution 3: SOTA among inference-time multi-agent methods on hard benchmarks

**Section 2: Related Work** (1.5p)
- Multi-agent debate/collaboration, in-context learning, evolutionary LLM agents, self-consistency

**Section 3: Method** (3p)
- 3.1 Framework overview (pool, team selection, experience buffer)
- 3.2 Experience accumulation (how agents store and retrieve strategies)
- 3.3 Collaborative reflection (CoDream and LeadLearn as TWO instantiations — equal billing)
- 3.4 Evolutionary lifecycle (fork, merge, retire, diversity)
- 3.5 Algorithm pseudocode

**Section 4: Experimental Setup** (1.5p)
- Benchmarks, baselines, ablations, 3 seeds, randomized ordering, metrics

**Section 5: Results** (3p)
- 5.1 Main results (mean±std, 3 seeds)
- 5.2 Learning curves (randomized ordering — THE key figure)
- 5.3 Ablation waterfall: SA → +experience → +lifecycle → +reflection
- 5.4 Code results (complete)

**Section 6: Analysis** (2p)
- 6.1 What do agents learn? (qualitative examples)
- 6.2 Population dynamics (diversity over time)
- 6.3 When does reflection help? (difficulty-stratified)
- 6.4 Computational cost
- 6.5 Sensitivity (pool size, team size)

**Section 7: Discussion & Limitations** (1p)
**Section 8: Conclusion** (0.5p)

---

## PRIORITY LIST

| Priority | Action | Impact on Score |
|----------|--------|----------------|
| 1 | Run 3 seeds on top-5 methods | 4→5.5 (rigorous) |
| 2 | Randomize math task ordering | 5.5→6 (clean learning curve) |
| 3 | Reframe narrative (framework, not single method) | 6→6.5 (defensible) |
| 4 | Complete code results | 6.5→6.5 (removes asterisk) |
| 5 | Compute-matched SC | 6.5→7 (strongest comparison) |
| 6 | Qualitative analysis | 7→7.5 (makes CoDream tangible) |

Expected final score with all changes: **6.5–7.5** (Weak Accept to Accept)
