#!/bin/bash
#SBATCH -J evopool_e5l
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 5:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --array=0-3%4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e5l_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e5l_%A_%a.err

# E5l: MATH fraction normalization fix + explicit LaTeX boxed prompt
CONDITIONS=("evopool_no_codream" "evopool_full")
SEEDS=(42 456)

N_COND=${#CONDITIONS[@]}
COND_IDX=$(( SLURM_ARRAY_TASK_ID % N_COND ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_COND ))

COND=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== EvoPool E5l (MATH fraction fix + LaTeX boxed prompt) ==="
echo "Job: ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | $(date)"
echo "Condition: $COND | Seed: $SEED"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code
export EVOPOOL_LOCAL_LLM_URL="http://10.217.117.45:8000"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e5l

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark aflow_stream \
    --n_per_domain 10 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed "$SEED" \
    --output_dir results/e5l/

echo "=== Done: $COND seed=$SEED | $(date) ==="
