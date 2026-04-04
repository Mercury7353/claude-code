#!/bin/bash
#SBATCH -J evopool_e7
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-2%3
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e7_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e7_%A_%a.err

# E7: Training-free open-source baselines at 100 tasks/domain
# self_consistency (k=5) | dylan | aflow
# Single seed=42, same setup as E6 for fair comparison

CONDITIONS=("self_consistency" "dylan" "aflow")
SEED=42

COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "=== EvoPool E7 (training-free baselines, 100 tasks/domain) ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | $(date)"
echo "Condition: $COND | Seed: $SEED"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

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
mkdir -p results/e7

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e7/

echo "=== Done: $COND seed=$SEED | $(date) ==="
