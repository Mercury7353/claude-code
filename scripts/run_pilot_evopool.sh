#!/bin/bash
#SBATCH -J evopool_pilot
#SBATCH -A hw-grp
#SBATCH -p dgxh
#SBATCH --gres=gpu:0
#SBATCH -t 2:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --array=0-1
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/pilot2_%A_%a.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/pilot2_%A_%a.err
#SBATCH --mail-type=END,FAIL

CONDITIONS=("evopool_full" "evopool_no_codream")
COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "=== EvoPool Pilot (retry): $COND ==="
echo "Job: $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID | Node: $SLURMD_NODENAME | Start: $(date)"

# Activate conda env
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

cd /nfs/hpc/share/zhanyaol/claude-code
export EVOPOOL_LOCAL_LLM_URL="http://10.217.117.45:8000"
mkdir -p results/pilot

python -u run_experiment.py \
    --condition "$COND" \
    --benchmark gsm8k_stream \
    --n_tasks 30 \
    --pool_size 10 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/pilot/

echo "=== Done: $COND | $(date) ==="
