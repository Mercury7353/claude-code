#!/bin/bash
#SBATCH -J evopool_pilot
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o logs/pilot_%j.out
#SBATCH -e logs/pilot_%j.err
#SBATCH --mail-type=END,FAIL

echo "=== EvoPool Pilot Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# Environment setup
module load cuda/12.1 2>/dev/null || true
module load python/3.10 2>/dev/null || true

cd /nfs/hpc/share/zhanyaol/claude-code

# Install dependencies if needed
pip install -q anthropic datasets pandas 2>/dev/null || true

mkdir -p logs results/pilot

# Run pilot: 3 conditions on GSM8K stream (30 tasks, 1 seed)
CONDITIONS=("evopool_cod_only" "evopool_no_codream" "no_memory")

for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo ">>> Running condition: $COND"
    python run_experiment.py \
        --condition "$COND" \
        --benchmark gsm8k_stream \
        --n_tasks 30 \
        --pool_size 10 \
        --team_size 3 \
        --backbone_llm "claude-sonnet-4-6" \
        --seed 42 \
        --output_dir results/pilot/
    echo ">>> Done: $COND"
done

echo ""
echo "=== Pilot Complete ==="
echo "End: $(date)"

# Quick comparison
python -c "
import json, glob
files = glob.glob('results/pilot/*.json')
for f in sorted(files):
    with open(f) as fp:
        d = json.load(fp)
    s = d['summary']
    print(f'{s[\"system\"]:30s} mean={s[\"mean_score\"]:.3f} final={s[\"final_score\"]:.3f} slope={s[\"learning_slope\"]:.4f}')
"
