#!/bin/bash
# Quick check of all hard math stream experiment progress

LOGS="/nfs/hpc/share/zhanyaol/claude-code/logs"
RESULTS="/nfs/hpc/share/zhanyaol/claude-code/results"

echo "=== Experiment Progress Monitor ==="
echo "$(date)"
echo ""

print_exp() {
    local name="$1"
    local logfile="$2"
    local resultfile="$3"

    if [ -f "$resultfile" ] && [ $(stat -c%s "$resultfile") -gt 5000 ]; then
        # Result file complete
        tasks=$(python3 -c "import json; d=json.load(open('$resultfile')); print(len(d.get('per_task_results',[])))" 2>/dev/null)
        mean=$(python3 -c "import json; d=json.load(open('$resultfile')); print(f\"{d['summary']['mean_score']:.3f}\")" 2>/dev/null)
        printf "%-20s ✅ DONE  tasks=%-4s mean=%s\n" "$name" "$tasks" "$mean"
    elif [ -f "$logfile" ]; then
        last=$(grep "Task.*Recent mean" "$logfile" 2>/dev/null | tail -1 | sed 's/.*Task \([0-9]*\).* Recent mean: \([0-9.]*\).*/\1 (\2)/')
        printf "%-20s 🔄 %-20s  %s\n" "$name" "$last" "$(ls -la $logfile | awk '{print $6, $7, $8}')"
    else
        printf "%-20s ⏳ queued\n" "$name"
    fi
}

# Single-agent (complete)
print_exp "E39 Single-Agent" "" "$RESULTS/e39/single_agent_hard_math_stream_seed42.json"

# EvoPool and noCoDream (key comparisons) - NEW JOB IDs (with mas.py fix)
print_exp "E40 EvoPool" "$LOGS/e40_evopool_aime_think_20144520.out" "$RESULTS/e40/evopool_full_hard_math_stream_seed42.json"
print_exp "E41 noCoDream" "$LOGS/e41_nocd_aime_think_20144521.out" "$RESULTS/e41/evopool_no_codream_hard_math_stream_seed42.json"

# Baselines
print_exp "E48 noLifecycle" "$LOGS/e48_nolifecycle_aime_20144522.out" "$RESULTS/e48/evopool_no_lifecycle_hard_math_stream_seed42.json"
print_exp "E49 noL2" "$LOGS/e49_nol2_aime_20144523.out" "$RESULTS/e49/evopool_no_l2_hard_math_stream_seed42.json"
print_exp "E50 noVerify" "$LOGS/e50_noverify_aime_20144524.out" "$RESULTS/e50/evopool_no_verify_hard_math_stream_seed42.json"

# Baselines (different code, not affected by MAS fix)
print_exp "E51 AgentNet" "$LOGS/e51_agentnet_aime_think_20144525.out" "$RESULTS/e51/agentnet_hard_math_stream_seed42.json"
print_exp "E52 MemCollab" "$LOGS/e52_memcollab_aime_think2_20144526.out" "$RESULTS/e52/memcollab_hard_math_stream_seed42.json"
print_exp "E53 EvoMem" "$LOGS/e53_evomem_aime_think2_20144527.out" "$RESULTS/e53/evomem_hard_math_stream_seed42.json"
print_exp "E54 DyLAN(fix)" "$LOGS/e54_dylan_aime_think_20141861.out" "$RESULTS/e54/dylan_hard_math_stream_seed42.json"
print_exp "E42 DyLAN(old)" "" "$RESULTS/e42/dylan_hard_math_stream_seed42.json"

# Order ablations
print_exp "E55 Reverse" "$LOGS/e55_reverse_order_20144616.out" "$RESULTS/e55/evopool_full_hard_math_stream_seed42.json"
print_exp "E56 Shuffled" "$LOGS/e56_shuffled_20144617.out" "$RESULTS/e56/evopool_full_hard_math_stream_seed42.json"
print_exp "E57 AIME-only" "$LOGS/e57_aime_only_20144618.out" "$RESULTS/e57/evopool_full_hard_math_stream_seed42.json"

echo ""
echo "=== Partial AIME results (from logs) ==="
python3 - << 'PYEOF'
import subprocess, re, os

def get_bins_from_log(logpath):
    try:
        with open(logpath) as f:
            content = f.read()
        bins = re.findall(r'Task (\d+)/135 \| Recent mean: ([0-9.]+)', content)
        return [(int(t), float(m)) for t, m in bins]
    except:
        return []

BASE = "/nfs/hpc/share/zhanyaol/claude-code/logs"

exps = [
    ("E40 EvoPool(fix)", f"{BASE}/e40_evopool_aime_think_20144520.out"),
    ("E41 noCoDream(fix)", f"{BASE}/e41_nocd_aime_think_20144521.out"),
    ("E48 noLifecycle(fix)", f"{BASE}/e48_nolifecycle_aime_20144522.out"),
    ("E49 noL2(fix)", f"{BASE}/e49_nol2_aime_20144523.out"),
    ("E50 noVerify(fix)", f"{BASE}/e50_noverify_aime_20144524.out"),
    ("E51 AgentNet", f"{BASE}/e51_agentnet_aime_think_20144525.out"),
    ("E52 MemCollab", f"{BASE}/e52_memcollab_aime_think2_20144526.out"),
    ("E53 EvoMem", f"{BASE}/e53_evomem_aime_think2_20144527.out"),
    ("E54 DyLAN(fix)", f"{BASE}/e54_dylan_aime_think_20141861.out"),
    ("E55 Reverse", f"{BASE}/e55_reverse_order_20144616.out"),
    ("E56 Shuffled", f"{BASE}/e56_shuffled_20144617.out"),
    ("E57 AIME-only", f"{BASE}/e57_aime_only_20144618.out"),
]

def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)

print(f"{'Exp':<22} {'MH-B1':>6} {'MH-B2':>6} {'MH-B3':>6} | {'A22-B1':>7} {'A22-B2':>7} {'A22-B3':>7} | {'A23-B1':>7}")
print("-" * 88)

for name, logpath in exps:
    if not os.path.exists(logpath):
        continue
    bins = get_bins_from_log(logpath)
    if not bins:
        continue
    b = {t: m for t, m in bins}
    mh1 = b.get(10, '—')
    mh2 = b.get(20, '—')
    mh3 = b.get(30, '—')
    a22_1 = b.get(40, '—')
    a22_2 = b.get(50, '—')
    a22_3 = b.get(60, '—')
    a23_1 = b.get(70, '—')

    print(f"{name:<22} {fmt(mh1):>6} {fmt(mh2):>6} {fmt(mh3):>6} | {fmt(a22_1):>7} {fmt(a22_2):>7} {fmt(a22_3):>7} | {fmt(a23_1):>7}")

PYEOF
