#!/bin/bash
#SBATCH -J val_math
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/val_math_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/val_math_%j.err

echo "=== Validate math fixes: EvoPool v3 vs SA on math_hard 30 ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URLS=""
check_server() { curl -s --max-time 5 "${1}/v1/models" | grep -q qwen 2>/dev/null; }
for port in 8007 8008; do
    candidate="http://10.217.117.45:${port}"
    if check_server "$candidate"; then
        [ -n "$URLS" ] && URLS="${URLS},"
        URLS="${URLS}${candidate}"
        echo "Server: $candidate"
    fi
done
[ -z "$URLS" ] && echo "ERROR: No vLLM" && exit 1
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1

echo "=== EvoPool v3 (max_tokens=7000 + retry + exp transfer) on math_hard 30 ==="
mkdir -p results/val_evopool_math30
python -u run_experiment.py \
    --condition "evopool_full" \
    --benchmark hard_math_stream \
    --n_per_domain 30 \
    --domains "math_hard" \
    --pool_size 20 --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/val_evopool_math30/

echo "=== SA baseline on math_hard 30 ==="
mkdir -p results/val_sa_math30
python -u run_experiment.py \
    --condition "single_agent" \
    --benchmark hard_math_stream \
    --n_per_domain 30 \
    --domains "math_hard" \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/val_sa_math30/

echo "=== SC baseline on math_hard 30 ==="
mkdir -p results/val_sc_math30
python -u run_experiment.py \
    --condition "self_consistency" \
    --benchmark hard_math_stream \
    --n_per_domain 30 \
    --domains "math_hard" \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/val_sc_math30/

echo "=== Done | $(date) ==="
