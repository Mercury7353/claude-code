#!/bin/bash
#SBATCH -J aflow_seed789
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 3:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/aflow_seed789_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/aflow_seed789_%j.err

echo "=== AFlow seed789 baseline ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURMD_NODENAME} | $(date)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool

cd /nfs/hpc/share/zhanyaol/claude-code
export EVOPOOL_LOCAL_LLM_URL="http://10.217.117.45:8000"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e5

python -u run_experiment.py \
    --condition "aflow" \
    --benchmark aflow_stream \
    --n_per_domain 10 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 789 \
    --output_dir results/e5/

echo "=== Done: aflow seed=789 | $(date) ==="
