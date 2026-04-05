#!/bin/bash
# Post-E17 analysis checklist
# Run after results/e17/evopool_full_aflow_stream_seed42.json appears

E17_RESULT="results/e17/evopool_full_aflow_stream_seed42.json"
E18_RESULT="results/e18/single_agent_aflow_stream_seed42.json"

cd /nfs/hpc/share/zhanyaol/claude-code

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

echo "=== Post-E17 Analysis ==="
echo "Date: $(date)"

# Check which results are available
if [ -f "$E17_RESULT" ]; then
    echo "✅ E17 results available"
    python3 -c "
import json
d = json.load(open('$E17_RESULT'))
s = d['summary']
print(f'  E17 Mean: {s[\"mean_score\"]:.4f}')
print(f'  AUC: {s[\"auc\"]:.4f}')
ds = d['domain_scores']
for domain in ['gsm8k','hotpotqa','mbpp','math','humaneval','drop']:
    scores = ds.get(domain, [])
    if scores:
        m = sum(scores)/len(scores)
        print(f'  {domain}: {m:.3f} ({len(scores)} tasks)')

# Detect potential server-outage 0.0 clusters (DROP domain = tasks 501-600)
per_task = d.get('per_task_results', [])
drop_tasks = [(r['task_index'], r['score']) for r in per_task if r.get('domain') == 'drop']
if drop_tasks:
    zero_drop = [t for t, s in drop_tasks if s == 0.0]
    if zero_drop:
        print(f'  ⚠️  DROP tasks with 0.0 score: {len(zero_drop)} tasks at indices {zero_drop[:10]}...')
        # Check if they cluster at end (server outage pattern)
        if max(zero_drop) >= 90:
            print('  ⚠️  0.0 cluster at end of DROP — possible server expiry gap')
    else:
        print('  ✅ No DROP zero-score cluster detected')
"
else
    echo "⚠️  E17 not ready yet: $E17_RESULT"
fi

if [ -f "$E18_RESULT" ]; then
    echo "✅ E18 results available"
else
    echo "⏳ E18 not ready yet"
fi

echo ""
echo "=== Summary Table ==="
python3 scripts/analyze_results.py

echo ""
echo "=== Generating Paper Figures ==="
python3 figures/plot_results.py

echo ""
echo "=== Post-E17 TODO ==="
echo "1. Fill TBD values in MyContext/research/paper_experiments_draft.md"
echo "2. Fill TBD values in MyContext/research/paper_intro_draft.md"
echo "3. Fill TBD values in MyContext/research/direction.md"
echo "4. Submit E19 lifecycle ablation: sbatch scripts/run_e19_no_lifecycle.sh"
echo "5. Push updated figures to both repos"
echo "6. Start LaTeX paper writing"
