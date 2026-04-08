#!/bin/bash
# Experiment watchdog: checks every 5 min, resubmits stale jobs, auto-submits follow-ups
LOG="/nfs/hpc/share/zhanyaol/claude-code/logs/experiment_watchdog.log"
cd /nfs/hpc/share/zhanyaol/claude-code

STALE_THRESHOLD=1800  # 30 minutes (EvoPool v2 + CoDream + thinking can take 15+ min/task)

while true; do
    now=$(date +%s)
    echo "[$(date)] Checking experiments..." >> "$LOG"

    # Check each running job for staleness
    squeue -u zhanyaol --noheader -o "%i %j %T" 2>/dev/null | grep -v vllm | while read jid name state; do
        [ "$state" != "RUNNING" ] && continue
        logfile=$(ls -t logs/*_${jid}.out 2>/dev/null | head -1)
        [ -f "$logfile" ] || continue
        mod=$(stat -c %Y "$logfile" 2>/dev/null)
        age=$(( now - mod ))
        if [ $age -gt $STALE_THRESHOLD ]; then
            expname=$(basename "$logfile" | sed "s/_${jid}.out//")
            echo "[$(date)] STALE: $expname (job $jid) - log ${age}s old. Resubmitting..." >> "$LOG"
            scancel "$jid" 2>/dev/null
            script="scripts/run_${expname}.sh"
            if [ -f "$script" ]; then
                sleep 5
                sbatch "$script" >> "$LOG" 2>&1
            fi
        fi
    done

    # Check vLLM health
    for port in 8007 8008; do
        if ! curl -s --max-time 5 "http://10.217.117.45:${port}/v1/models" 2>/dev/null | grep -q qwen; then
            echo "[$(date)] vLLM $port DOWN - restarting" >> "$LOG"
            sbatch "scripts/serve_vllm_${port}.sh" >> "$LOG" 2>&1
            sleep 180
        fi
    done

    # Auto-submit ARC-AGI-3 SA baseline when EvoPool finishes
    if [ -f "results/arc3_gpt54/evopool_full_arc_agi3_seed42.json" ] && \
       [ ! -f "results/arc3_gpt54_sa/.submitted" ]; then
        echo "[$(date)] ARC-AGI-3 EvoPool done! Submitting SA baseline..." >> "$LOG"
        sbatch scripts/run_arc3_sa_gpt54.sh >> "$LOG" 2>&1
        touch results/arc3_gpt54_sa/.submitted
    fi

    sleep 300
done
