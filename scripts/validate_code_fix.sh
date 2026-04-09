#!/bin/bash
#SBATCH -J val_code_fix
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/val_code_fix_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/val_code_fix_%j.err

echo "=== Validate code fix: EvoPool vs SA on MBPP (50 tasks) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

# Find live vLLM servers
URLS=""
check_server() { curl -s --max-time 5 "${1}/v1/models" | grep -q qwen 2>/dev/null; }
for port in 8007 8008; do
    candidate="http://10.217.117.45:${port}"
    if check_server "$candidate"; then
        [ -n "$URLS" ] && URLS="${URLS},"
        URLS="${URLS}${candidate}"
        echo "Found server: $candidate"
    fi
done
[ -z "$URLS" ] && echo "ERROR: No vLLM servers" && exit 1
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1

echo ""
echo "=== Run 1: EvoPool (with test case fix) on MBPP 50 ==="
mkdir -p results/val_evopool_mbpp50
python -u run_experiment.py \
    --condition "evopool" \
    --benchmark hard_code_stream \
    --n_per_domain 50 \
    --domains "mbpp" \
    --pool_size 20 --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/val_evopool_mbpp50/

echo ""
echo "=== Run 2: SA baseline on MBPP 50 ==="
mkdir -p results/val_sa_mbpp50
python -u run_experiment.py \
    --condition "single_agent" \
    --benchmark hard_code_stream \
    --n_per_domain 50 \
    --domains "mbpp" \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/val_sa_mbpp50/

echo ""
echo "=== Run 3: SC baseline on MBPP 50 ==="
mkdir -p results/val_sc_mbpp50
python -u run_experiment.py \
    --condition "self_consistency" \
    --benchmark hard_code_stream \
    --n_per_domain 50 \
    --domains "mbpp" \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/val_sc_mbpp50/

echo ""
echo "=== All validation runs complete | $(date) ==="
