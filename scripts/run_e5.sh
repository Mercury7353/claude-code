#!/bin/bash
#SBATCH -J evopool_e5
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 5:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --array=0-11%5
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e5_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e5_%A_%a.err

# E5: ALL fixes — definitive experiment
# - Fork suppression at domain boundary (semantic embeddings)
# - Co-Dream 2-phase score-gated + domain constraint
# - Code format instructions for HumanEval/MBPP
# - CORRECT HumanEval evaluation (check() actually called)
# - DyLAN rounds fix
# - AFlow entry_point fix
# 4 conditions × 3 seeds = 12 jobs
CONDITIONS=(
    "evopool_full"        # all fixes
    "evopool_no_codream"  # ablation: no Co-Dream
    "dylan"               # baseline (rounds=3 fix)
    "aflow"               # baseline (entry_point fix + correct eval)
)
SEEDS=(42 123 456)

N_COND=${#CONDITIONS[@]}
COND_IDX=$(( SLURM_ARRAY_TASK_ID % N_COND ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_COND ))

COND=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== EvoPool E5 (definitive) ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: ${SLURMD_NODENAME} | $(date)"
echo "Condition: $COND | Seed: $SEED"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

cd /nfs/hpc/share/zhanyaol/claude-code
export EVOPOOL_LOCAL_LLM_URL="http://10.217.117.45:8000"
export HF_DATASETS_OFFLINE=1  # Use local cache to avoid transient network failures
mkdir -p results/e5

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 10 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e5/

echo "=== Done: $COND seed=$SEED | $(date) ==="
