#!/bin/bash
#SBATCH -J evopool_e2
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 4:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --array=0-11%5
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e2_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e2_%A_%a.err

# E2: Re-run with all fixes applied
# 4 key conditions × 3 seeds = 12 jobs
CONDITIONS=(
    "evopool_full"        # main method (fork fix + codream fix)
    "evopool_no_codream"  # ablation: no codream
    "dylan"               # baseline (sacrebleu fix)
    "aflow"               # baseline (entry_point fix)
)
SEEDS=(42 123 456)

N_COND=${#CONDITIONS[@]}
COND_IDX=$(( SLURM_ARRAY_TASK_ID % N_COND ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_COND ))

COND=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== EvoPool E2 ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: ${SLURMD_NODENAME} | $(date)"
echo "Condition: $COND | Seed: $SEED"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

cd /nfs/hpc/share/zhanyaol/claude-code
export EVOPOOL_LOCAL_LLM_URL="http://10.217.117.45:8000"
mkdir -p results/e2

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 10 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e2/

echo "=== Done: $COND seed=$SEED | $(date) ==="
