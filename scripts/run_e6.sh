#!/bin/bash
#SBATCH -J evopool_e6
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-2%3
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e6_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e6_%A_%a.err

# E6: Large-scale 100 tasks/domain experiment
# Conditions: evopool_full | evopool_no_codream | aflow_baseline
# Single seed=42 (100 tasks/domain makes noise irrelevant)
# Uses all available vLLM servers for load balancing

CONDITIONS=("evopool_full" "evopool_no_codream" "aflow")
SEED=42

COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "=== EvoPool E6 (100 tasks/domain, all fixes, new CoDream) ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | $(date)"
echo "Condition: $COND | Seed: $SEED | N_per_domain: 100"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

# Wait for at least one vLLM server to be ready (up to 20 min)
echo "Waiting for vLLM server(s) to be ready..."
for i in $(seq 1 40); do
    for f in vllm_server.json vllm_server_2.json vllm_server_3.json; do
        if [ -f "$f" ]; then
            URL=$(python3 -c "import json; d=json.load(open('$f')); print(d['url'])" 2>/dev/null)
            if [ -n "$URL" ]; then
                # Check if server responds
                if python3 -c "import requests; requests.get('$URL/health', timeout=5)" 2>/dev/null; then
                    break 2
                fi
            fi
        fi
    done
    echo "  Waiting... (${i}/40, $(date))"
    sleep 30
done

# Collect all available vLLM server URLs
URLS=""
for f in vllm_server.json vllm_server_2.json vllm_server_3.json; do
    if [ -f "$f" ]; then
        URL=$(python3 -c "import json; d=json.load(open('$f')); print(d['url'])" 2>/dev/null)
        if [ -n "$URL" ]; then
            if [ -n "$URLS" ]; then URLS="$URLS,$URL"; else URLS="$URL"; fi
        fi
    fi
done

if [ -z "$URLS" ]; then
    echo "ERROR: No vLLM server JSON files found"
    exit 1
fi
echo "Using servers: $URLS"
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e6

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e6/

echo "=== Done: $COND seed=$SEED | $(date) ==="
