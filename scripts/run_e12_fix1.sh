#!/bin/bash
#SBATCH -J evopool_e12_fix1
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 48:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e12_fix1_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e12_fix1_%j.err

# E12: EvoPool full + Fix1 (scope-tagged Co-Dream insights)
# Fix 1: subdomain insights stored in scoped dict, only injected when task matches.
# Prevents algebra insights from polluting geometry/combinatorics agents.
echo "=== EvoPool E12 fix1 (scope-tagged Co-Dream) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URLS=""
for f in vllm_server_4.json vllm_server_5.json; do
    if [ -f "$f" ]; then
        URL=$(python3 -c "import json; d=json.load(open('$f')); print(d['url'])" 2>/dev/null)
        [ -n "$URL" ] && URLS="${URLS}${URLS:+,}${URL}"
    fi
done
if [ -z "$URLS" ]; then
    echo "ERROR: No long-lived servers found (need vllm_server_4.json / vllm_server_5.json)"
    exit 1
fi
echo "Using servers: $URLS"
export EVOPOOL_LOCAL_LLM_URLS="$URLS"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e12

python -u run_experiment.py \
    --condition "evopool_full" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e12/

echo "=== Done: evopool_full fix1 | $(date) ==="
