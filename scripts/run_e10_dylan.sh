#!/bin/bash
#SBATCH -J evopool_e10_dylan
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e10_dylan_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e10_dylan_%j.err

# E10: DyLAN rerun with long-lived servers (4/5 on ports 8003/8004)
echo "=== EvoPool E10 DyLAN (100 tasks/domain) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URLS=""
for f in vllm_server_4.json vllm_server_5.json vllm_server_2.json vllm_server_3.json; do
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
mkdir -p results/e10

python -u run_experiment.py \
    --condition "dylan" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e10/

echo "=== Done: dylan | $(date) ==="
