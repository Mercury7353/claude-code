#!/bin/bash
# Overnight monitoring — polls every 20 min, appends to log
LOG="/nfs/hpc/share/zhanyaol/claude-code/logs/overnight_monitor.log"
echo "=== Overnight monitor started $(date) ===" >> "$LOG"

while true; do
    bash /nfs/hpc/share/zhanyaol/claude-code/scripts/check_progress.sh 2>/dev/null >> "$LOG"
    echo "---" >> "$LOG"

    # Check if E40 and all key experiments are done
    done=$(python3 -c "
import json, os
files = {
    'E40': 'results/e40/evopool_full_hard_math_stream_seed42.json',
    'E41': 'results/e41/evopool_no_codream_hard_math_stream_seed42.json',
    'E50': 'results/e50/evopool_no_verify_hard_math_stream_seed42.json',
    'E51': 'results/e51/agentnet_hard_math_stream_seed42.json',
    'E52': 'results/e52/memcollab_hard_math_stream_seed42.json',
}
os.chdir('/nfs/hpc/share/zhanyaol/claude-code')
all_done = all(os.path.exists(f) and os.path.getsize(f) > 5000 for f in files.values())
print('YES' if all_done else 'NO')
" 2>/dev/null)

    if [ "$done" = "YES" ]; then
        echo "=== All key experiments complete! $(date) ===" >> "$LOG"
        # Run full analysis
        python3 scripts/analyze_results.py >> "$LOG" 2>/dev/null
        break
    fi

    sleep 1200  # 20 min
done
