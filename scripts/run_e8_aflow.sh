#!/bin/bash
#SBATCH -J evopool_e8_aflow
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-1%2
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e8_aflow_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e8_aflow_%A_%a.err

# E8: AFlow rerun with dynamic server URL fix (DROP was 0 due to hardcoded dead server)
# Also run self_consistency (E7 aflow slot replacement) as E8.1
CONDITIONS=("aflow" "self_consistency")
SEED=42
COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "=== EvoPool E8 (AFlow fix + SC, 100 tasks/domain) ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | $(date)"
echo "Condition: $COND | Seed: $SEED"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URLS=""
for f in vllm_server.json vllm_server_2.json vllm_server_3.json; do
    if [ -f "$f" ]; then
        URL=$(python3 -c "import json; d=json.load(open('$f')); print(d['url'])" 2>/dev/null)
        if [ -n "$URL" ]; then
            if [ -n "$URLS" ]; then URLS="$URLS,$URL"; else URLS="$URL"; fi
        fi
    fi
done
echo "Using servers: $URLS"
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e8

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e8/

echo "=== Done: $COND | $(date) ==="
