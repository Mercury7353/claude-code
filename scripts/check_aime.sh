#!/bin/bash
cd /nfs/hpc/share/zhanyaol/claude-code
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool 2>/dev/null

python3 -c "
import json, os, sys

found_aime = False
for name, path in [('EvoPool', 'results/v2_hardmath'), ('noCoDream', 'results/v2_hardmath_nocd'), ('LeaderLearn', 'results/v2_hardmath_leadlearn')]:
    f = os.path.join(path, '_checkpoint.json')
    if not os.path.exists(f): continue
    data = json.load(open(f))
    tasks = data.get('per_task_results', [])
    n = len(tasks)
    
    aime_tasks = [t for t in tasks if t.get('domain','').startswith('aime')]
    if aime_tasks:
        found_aime = True
        aime_scores = [t.get('team_score', t.get('score', 0)) for t in aime_tasks]
        math_hard = [t for t in tasks if t.get('domain') == 'math_hard']
        mh_scores = [t.get('team_score', t.get('score', 0)) for t in math_hard]
        
        # AIME by year
        years = {}
        for t in aime_tasks:
            yr = t.get('domain', 'unknown')
            if yr not in years: years[yr] = []
            years[yr].append(t.get('team_score', t.get('score', 0)))
        
        print(f'=== {name}: AIME REACHED ({len(aime_tasks)} tasks) ===')
        print(f'  math_hard: {sum(mh_scores)/len(mh_scores):.4f} ({len(mh_scores)} tasks)')
        print(f'  AIME overall: {sum(aime_scores)/len(aime_scores):.4f}')
        for yr, scores in sorted(years.items()):
            print(f'  {yr}: {sum(scores)/len(scores):.4f} ({len(scores)} tasks)')
        
        # Collapse comparison
        mh_avg = sum(mh_scores)/len(mh_scores)
        aime_avg = sum(aime_scores)/len(aime_scores)
        collapse = (mh_avg - aime_avg) / mh_avg * 100
        print(f'  Collapse: {collapse:.1f}%')
        print()

if not found_aime:
    print('No experiments have reached AIME yet.')
    sys.exit(1)
" 2>/dev/null
