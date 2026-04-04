#!/bin/bash
#SBATCH -J evopool_main
#SBATCH -A hw-grp
#SBATCH -p dgxh
#SBATCH --gres=gpu:0
#SBATCH -t 3:30:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --array=0-23%5
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/main_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/main_%A_%a.err
#SBATCH --mail-type=END,FAIL

# 8 conditions × 3 seeds = 24 jobs (indices 0-23)
# Max 5 concurrent to avoid saturating vLLM server
CONDITIONS=(
    "evopool_full"
    "evopool_no_codream"
    "evopool_symmetric_codream"
    "evopool_no_lifecycle"
    "evopool_no_collab_score"
    "dylan"
    "agentnet"
    "aflow"
)
SEEDS=(42 123 456)

N_COND=${#CONDITIONS[@]}   # 7
COND_IDX=$(( SLURM_ARRAY_TASK_ID % N_COND ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_COND ))

COND=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== EvoPool Main Experiment ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: ${SLURMD_NODENAME} | Start: $(date)"
echo "Condition: $COND | Seed: $SEED"

# Activate conda env
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

cd /nfs/hpc/share/zhanyaol/claude-code
export EVOPOOL_LOCAL_LLM_URL="http://10.217.117.45:8000"
mkdir -p results/main

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 10 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/main/

echo "=== Done: $COND seed=$SEED | $(date) ==="
