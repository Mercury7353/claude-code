#!/bin/bash
#SBATCH -J val_cc20
#SBATCH -A hw-grp
#SBATCH -p hw-grp,share,preempt
#SBATCH --gres=gpu:0
#SBATCH -t 6:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/val_cc20_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/val_cc20_%j.err

echo "=== val_cc20: CodeContests validation (20 tasks, with prompt fix) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URLS=""
check_server() { curl -s --max-time 5 "${1}/v1/models" | grep -q qwen 2>/dev/null; }
for port in 8007 8008 8009 8010; do
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
mkdir -p results/val_cc20

python -u run_experiment.py \
    --condition "evopool_full" \
    --benchmark hard_code_stream \
    --n_per_domain 20 \
    --domains "code_contests" \
    --pool_size 20 --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/val_cc20/

echo "=== Done: val_cc20 | $(date) ==="
