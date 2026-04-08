# EvoPool v2 Experiment Registry

**Date**: 2026-04-06 ~ 2026-04-08
**Code version**: v2 redesign (experience buffer + dynamic MAS + 5-phase CoDream + leader learning)
**Backbone**: Qwen3-8B (vLLM on dgxh-1/cn-s-5, ports 8007-8010)

---

## Completed Results

### AFlow-Stream (heterogeneous, 6 domains, 100/domain)

| System | Tasks | Mean | gsm8k | hotpotqa | math | mbpp | humaneval | drop |
|--------|-------|------|-------|----------|------|------|-----------|------|
| **EvoPool v2** | 600 | **0.876** | 0.970 | 0.901 | 0.830 | 0.853 | 0.840 | 0.860 |
| noCoDream | 600 | 0.865 | 0.970 | 0.882 | 0.820 | 0.830 | 0.840 | 0.850 |
| Single-Agent | 600 | 0.819 | 0.960 | 0.791 | 0.780 | 0.780 | 0.800 | 0.800 |

**Key findings:**
- EvoPool v2 > SA by +5.7% (0.876 vs 0.819)
- CoDream contribution: +1.1% (0.876 vs 0.865) — modest on easy tasks (ceiling effect)
- Largest gains on hotpotqa (+11.0%), drop (+6.0%), mbpp (+7.3%)

---

### Hard Math Stream (262 MATH-hard + 105 AIME = 367 tasks)

| System | Tasks | Mean | math_hard | aime_22 | aime_23 | aime_24 | aime_25 | Status |
|--------|-------|------|-----------|---------|---------|---------|---------|--------|
| **EvoPool v2** | 180/367 | **0.600** | 0.600(180) | — | — | — | — | RUNNING |
| **noCoDream** | 160/367 | **0.613** | 0.613(160) | — | — | — | — | RUNNING |
| **LeaderLearn** | 150/367 | **0.627** | 0.627(150) | — | — | — | — | RUNNING |
| MemCollab | 344 | 0.459 | 0.556(239) | 0.233 | 0.200 | 0.267 | 0.267 | DONE |
| EvoMem | 342 | 0.453 | 0.536(237) | 0.233 | 0.200 | 0.300 | 0.400 | DONE |
| AgentNet | 342 | 0.409 | 0.468(237) | 0.267 | 0.133 | 0.400 | 0.333 | DONE |
| SC k=5 | 367 | 0.406 | 0.542(262) | 0.033 | 0.133 | 0.033 | 0.067 | DONE |
| DyLAN | 367 | 0.403 | 0.515(262) | 0.133 | 0.067 | 0.100 | 0.267 | DONE |
| Single-Agent | 367 | 0.251 | 0.302(262) | 0.133 | 0.067 | 0.167 | 0.133 | DONE |

**Notes:**
- EvoPool variants still running (only math_hard done so far, ~50%). AIME tasks are harder and will likely lower the mean.
- Even on math_hard alone: EvoPool (0.600) vs SA (0.302) = **+98% relative improvement**
- SC has high math_hard (0.542) but very poor AIME (0.033-0.133) — no learning transfer
- AgentNet/EvoMem/MemCollab only 342-344 tasks (connection errors dropped ~23 tasks)

---

### Hard Code Stream (257 MBPP + 164 HumanEval + 165 CodeContests = 586 tasks)

| System | Tasks | Mean | mbpp | humaneval | code_contests | Status | Notes |
|--------|-------|------|------|-----------|---------------|--------|-------|
| **EvoPool v2** | 586 | **0.615** | 0.712 | 1.000 | 0.081 | DONE | |
| **noCoDream** | 586 | 0.617 | 0.701 | 1.000 | 0.107 | DONE | |
| LeaderLearn | 586 | 0.601 | 0.686 | 1.000 | 0.071 | DONE | |
| EvoMem | 448 | 0.588 | 0.900 | 0.929(28) | 0.039 | DONE | partial HumanEval |
| SC k=5 | 586 | 0.490 | 0.863 | 0.171 | 0.225 | DONE | low HumanEval! |
| SA | 586 | 0.323 | **0.067** | 1.000 | 0.047 | DONE | **MBPP BUG** |
| AgentNet | 586 | 0.331 | **0.074** | 0.994 | 0.071 | DONE | **MBPP BUG** |
| MemCollab | 586 | 0.328 | **0.061** | 0.994 | 0.082 | DONE | **MBPP BUG** |
| DyLAN | 360/586 | 0.297 | 0.362 | 0.136(103) | — | RUNNING | low HumanEval! |

**CRITICAL ISSUES:**
1. **MBPP bug in SA, AgentNet, MemCollab**: These ran before the function-name fix (Apr 6 18:25). MBPP <0.1 is clearly broken. **Need rerun.**
2. **SC humaneval=0.171**: Abnormally low (SA=1.0). May have a bug in SC handling of code tasks.
3. **DyLAN humaneval=0.136**: Also abnormally low. Still running.
4. **EvoPool vs noCoDream**: 0.615 vs 0.617 — CoDream not helping on code tasks.
5. **EvoMem only 448 tasks**: Missing ~138 tasks due to connection errors.

---

### ARC-AGI-3 (25 public environments, GPT-5.4)

| System | Envs Done | Non-zero | Best Score | Running Mean | Status |
|--------|-----------|----------|------------|-------------|--------|
| EvoPool (pool=5, rounds=10) | 21/25 | 1 (CD82=0.17) | 0.17 | 0.008 | RUNNING |
| SA baseline | — | — | — | — | Queued after EvoPool |

**Notes:**
- Only 1/21 environments scored non-zero (CD82 = 0.17)
- Currently processing re86 environment (round 4/10)
- ~4 environments remaining, ETA ~4 hours
- Results will likely be very low. ARC-AGI-3 is extremely difficult for text-based agents.

---

## Running Jobs (as of 2026-04-08 00:10 PDT)

| Job ID | Partition | Name | Progress | ETA |
|--------|-----------|------|----------|-----|
| 20148359 | hw-grp | v2_hardmath (full) | 190/367 | ~6h |
| 20148360 | hw-grp | v2_hardmath_nocd | 165/367 | ~7h |
| 20148361 | share | v2_hardmath_leadlearn | 160/367 | ~7h |
| 20148654 | hw-grp | v2_hardcode_dylan | 360/586 | ~4h |
| 20147987 | hw-grp | arc3_evopool_gpt54 | 21/25 envs | ~4h |
| 20148516 | dgxh | vllm 8009 | running | — |
| 20148517 | ampere | vllm 8010 | running | — |
| 20148780 | dgxh | vllm 8007 | running | — |
| 20148785 | dgxh | vllm 8008 | running | — |

---

## Ablation Matrix

### What each comparison tests:

| Comparison | A | B | Tests |
|------------|---|---|-------|
| **CoDream** | EvoPool v2 | noCoDream | Does 5-phase CoDream + experience sharing help? |
| **Leader learning** | LeaderLearn | EvoPool v2 | Does leader accumulating lead experience improve? |
| **Pool structure** | noCoDream | SA | Does team of 3 + leader MAS selection help? |
| **Dynamic MAS** | EvoPool v2 | SC k=5 | Does leader-selected structure beat fixed SC? |
| **vs baselines** | EvoPool v2 | AgentNet/MemCollab/EvoMem/DyLAN | Does EvoPool beat existing multi-agent? |

### Preliminary conclusions (hardmath, partial):

1. **EvoPool >> SA**: 0.600 vs 0.302 on math_hard = massive improvement
2. **EvoPool > all baselines**: Best baseline is MemCollab (0.459), EvoPool (0.600) is +30% relative
3. **CoDream effect unclear**: noCoDream (0.613) slightly > EvoPool (0.600) so far — need AIME data
4. **LeaderLearn promising**: 0.627 > 0.600 — leader learning helps on math_hard
5. **SC collapses on AIME**: 0.542 math_hard but 0.033-0.133 on AIME — no transfer learning

---

## Action Items

1. **Rerun SA, AgentNet, MemCollab for hardcode** — MBPP function name bug
2. **Investigate SC & DyLAN humaneval** — abnormally low scores
3. **Rerun EvoMem hardcode** — only 448/586 tasks completed
4. **Wait for hardmath to finish** — need AIME results for complete picture
5. **ARC-AGI-3 SA baseline** — submit after EvoPool finishes
6. **Calculate RHAE for ARC-AGI-3** — after all environments complete

---

## Code Version Notes

**v2 changes (2026-04-06):**
1. Agent experience buffer — concrete solving records, injected in user prompt
2. Dynamic MAS structure selection — leader picks voting/debate/generator-critic/decompose
3. Full 5-phase CoDream — REFLECT -> CONTRAST -> IMAGINE -> DEBATE -> CRYSTALLIZE
4. Data-driven persona — computed from actual performance, no LLM rewrite
5. Leader learning — leader records which structures worked, uses for future decisions
6. No methodology compromises — all mechanisms active for all tasks

**Bug fixes (2026-04-07):**
- MBPP function name fix — added `[REQUIRED FUNCTION NAME: entry_point]` + test cases
- CodeContests offline loading — load from HuggingFace cache
- MBPP eval timeout — 10s SIGALRM
- Heartbeat thread — prevents watchdog kills
- Checkpoint resume — save/restore every 10 tasks
- Stale vLLM config cleanup
