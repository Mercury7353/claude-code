#!/bin/bash
#SBATCH -J hmath_v3nc
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/v2_hardmath_v3_nocd_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/v2_hardmath_v3_nocd_%j.err

echo "=== v2_hardmath_v3_nocd: all fixes, no CoDream ==="
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
mkdir -p results/v2_hardmath_v3_nocd

python -u run_experiment.py \
    --condition "evopool_no_codream" \
    --benchmark hard_math_stream \
    --n_per_domain all \
    --domains "math_hard,aime_2022,aime_2023,aime_2024,aime_2025" \
    --pool_size 20 --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/v2_hardmath_v3_nocd/

echo "=== Done | $(date) ==="
