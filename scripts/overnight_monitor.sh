#!/bin/bash
cd /nfs/hpc/share/zhanyaol/claude-code
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool 2>/dev/null

echo ""
echo "## Check at $(date '+%Y-%m-%d %H:%M PDT')"
echo ""

echo "### Running Jobs"
squeue -u zhanyaol --format="%.10i %.15j %.8T %.10M" 2>/dev/null | head -25
echo ""

echo "### Hard Math Progress"
python3 -c "
import json, os
experiments = [
    ('EvoPool', 'results/v2_hardmath'),
    ('noCoDream', 'results/v2_hardmath_nocd'),
    ('LeaderLearn', 'results/v2_hardmath_leadlearn'),
    ('EvoPool_FIXED', 'results/v2_hardmath_fixed'),
    ('noCoDream_FIXED', 'results/v2_hardmath_nocd_fixed'),
]
for name, path in experiments:
    f = os.path.join(path, '_checkpoint.json')
    if not os.path.exists(f): continue
    data = json.load(open(f))
    tasks = data.get('per_task_results', [])
    n = len(tasks)
    if n == 0: continue
    scores = [t.get('team_score', t.get('score', 0)) for t in tasks]
    avg = sum(scores)/n
    domains = {}
    for t in tasks:
        d = t.get('domain', '?')
        if d not in domains: domains[d] = []
        domains[d].append(t.get('team_score', t.get('score', 0)))
    domain_str = ', '.join(f'{k}={sum(v)/len(v):.3f}({len(v)})' for k,v in sorted(domains.items()))
    in_aime = any(d.startswith('aime') for d in domains)
    done = ' **DONE**' if n >= 360 else ''
    print(f'- **{name}**: {n}/367, avg={avg:.4f} | {domain_str}{done}')
    if in_aime:
        aime_tasks = [t for t in tasks if t.get('domain','').startswith('aime')]
        aime_scores = [t.get('team_score', t.get('score', 0)) for t in aime_tasks]
        math_hard = [t for t in tasks if t.get('domain') == 'math_hard']
        mh_scores = [t.get('team_score', t.get('score', 0)) for t in math_hard]
        mh_avg = sum(mh_scores)/len(mh_scores) if mh_scores else 0
        aime_avg = sum(aime_scores)/len(aime_scores)
        collapse = (mh_avg - aime_avg) / mh_avg * 100 if mh_avg > 0 else 0
        print(f'  >> AIME: {len(aime_tasks)} tasks, avg={aime_avg:.4f}, collapse={collapse:.1f}%')
" 2>/dev/null
echo ""

echo "### Hard Code Reruns"
python3 -c "
import json, os
experiments = [
    ('SA', 'results/v2_hardcode_sa_rerun'),
    ('AgentNet', 'results/v2_hardcode_agentnet_rerun'),
    ('MemCollab', 'results/v2_hardcode_memcollab_rerun'),
    ('EvoMem', 'results/v2_hardcode_evomem_rerun'),
    ('DyLAN', 'results/v2_hardcode_dylan_rerun'),
    ('SC', 'results/v2_hardcode_sc_rerun'),
]
for name, path in experiments:
    f = os.path.join(path, '_checkpoint.json')
    if not os.path.exists(f): continue
    data = json.load(open(f))
    tasks = data.get('per_task_results', [])
    n = len(tasks)
    if n == 0: continue
    scores = [t.get('team_score', t.get('score', 0)) for t in tasks]
    avg = sum(scores)/n
    domains = {}
    for t in tasks:
        d = t.get('domain', '?')
        if d not in domains: domains[d] = []
        domains[d].append(t.get('team_score', t.get('score', 0)))
    domain_str = ', '.join(f'{k}={sum(v)/len(v):.3f}({len(v)})' for k,v in sorted(domains.items()))
    done = ' **DONE**' if n >= 580 else ''
    print(f'- **{name}**: {n}/586, avg={avg:.4f} | {domain_str}{done}')
" 2>/dev/null
echo ""
echo "---"
