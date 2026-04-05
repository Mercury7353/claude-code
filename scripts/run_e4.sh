#!/bin/bash
#SBATCH -J evopool_e4
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 5:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --array=0-8%4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e4_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e4_%A_%a.err

# E4: + code formatting fix (execute_subtask & synthesize prompt for HumanEval/MBPP)
# 3 conditions × 3 seeds = 9 jobs
CONDITIONS=(
    "evopool_full"         # all fixes including code format
    "evopool_no_codream"   # ablation: no codream, but with code format fix
    "aflow"                # baseline comparison
)
SEEDS=(42 123 456)

N_COND=${#CONDITIONS[@]}
COND_IDX=$(( SLURM_ARRAY_TASK_ID % N_COND ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_COND ))

COND=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== EvoPool E4 ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: ${SLURMD_NODENAME} | $(date)"
echo "Condition: $COND | Seed: $SEED"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

cd /nfs/hpc/share/zhanyaol/claude-code
export EVOPOOL_LOCAL_LLM_URL="http://10.217.117.45:8000"
mkdir -p results/e4

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 10 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e4/

echo "=== Done: $COND seed=$SEED | $(date) ==="
