#!/bin/bash
# Monitor all experiments - run periodically
cd /nfs/hpc/share/zhanyaol/claude-code
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool 2>/dev/null

echo "=== EvoPool Monitor Report $(date) ==="
echo ""

# Job status
echo "## Running Jobs"
squeue -u zhanyaol --format="%.10i %.12P %.15j %.2t %.10M" 2>/dev/null
echo ""

# Parse checkpoint results
echo "## Experiment Results"
python3 << 'PYEOF'
import json, os, glob

def parse_checkpoint(path):
    try:
        with open(path) as f:
            data = json.load(f)
        results = data.get('results', data.get('per_task_results', {}))
        if isinstance(results, dict):
            results = list(results.values())
        total = len(results)
        if total == 0:
            return None
        correct = sum(1 for r in results if r.get('score', r.get('correct', 0)) >= 0.5)
        domains = {}
        for r in results:
            d = r.get('domain', r.get('task_type', 'unknown'))
            if d not in domains:
                domains[d] = {'n': 0, 'c': 0}
            domains[d]['n'] += 1
            domains[d]['c'] += 1 if r.get('score', r.get('correct', 0)) >= 0.5 else 0
        return {'total': total, 'correct': correct, 'acc': correct/total, 'domains': domains}
    except:
        return None

# Check all result directories
for dirpath in sorted(glob.glob('results/v2_*') + glob.glob('results/val_*')):
    cp = os.path.join(dirpath, '_checkpoint.json')
    # Also check for final result files
    jsons = glob.glob(os.path.join(dirpath, '*.json'))
    jsons = [j for j in jsons if '_checkpoint' not in j]

    target = cp if os.path.exists(cp) else (jsons[0] if jsons else None)
    if target is None:
        continue

    r = parse_checkpoint(target)
    if r is None:
        continue

    name = os.path.basename(dirpath)
    print(f"\n{name}: {r['total']} tasks, Overall={r['acc']:.3f}")
    for d, v in sorted(r['domains'].items()):
        acc = v['c']/v['n'] if v['n'] > 0 else 0
        print(f"  {d}: {v['c']}/{v['n']} = {acc:.3f}")
PYEOF

echo ""
echo "=== End Report ==="
