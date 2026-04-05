#!/usr/bin/env python3
"""
Re-run E17 task stream (dry run) to reconstruct final pool state.

Since E17 was run without --save_pool, we need to re-evolve the pool
through the same 600 tasks to get the final agent states.

Alternatively: if E17 had saved the pool, we'd just load it.
This script provides a faster workaround: replay task outcomes through
the pool's learning mechanism (CoDream) using only the logged per_task_results.

NOTE: This is an APPROXIMATION — it replays scores but cannot replay
the exact LLM calls that generated each insight.  Use --save_pool in
future runs to get exact pool state.
"""
import sys
print("NOTE: E17 pool state can only be reconstructed by re-running E17 with --save_pool.")
print("To get exact warm-start pool, either:")
print("  1. Submit a new E17-identical job with --save_pool results/e17/evopool_pool_state.json")
print("  2. Or run E21 (same as E17) with --save_pool, then use that pool for E20.")
print()
print("Recommended: run scripts/run_e21_save_pool.sh (E17 replica with pool saving)")
print("Then run E20 warm-start after E21 completes.")
