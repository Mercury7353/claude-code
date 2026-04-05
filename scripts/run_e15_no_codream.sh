#!/bin/bash
#SBATCH -J evopool_e15_nocd
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 8:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e15_no_codream_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e15_no_codream_%j.err

# E15: EvoPool no-CoDream ablation (clean run with exec timeout fix)
# Ablation baseline for paper: shows benefit of CoDream over individual memory only.
# Uses servers 6/7 (expire ~1PM).
echo "=== EvoPool E15 no-CoDream ablation ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

URL6=""
URL7=""
[ -f "vllm_server_6.json" ] && URL6=$(python3 -c "import json; d=json.load(open('vllm_server_6.json')); print(d['url'])" 2>/dev/null)
[ -f "vllm_server_7.json" ] && URL7=$(python3 -c "import json; d=json.load(open('vllm_server_7.json')); print(d['url'])" 2>/dev/null)

if [ -z "$URL6" ] || [ -z "$URL7" ]; then
    echo "ERROR: Servers 6/7 not found"
    exit 1
fi
echo "Using servers: $URL6, $URL7"
export EVOPOOL_LOCAL_LLM_URLS="$URL6,$URL7"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e15

python -u run_experiment.py \
    --condition "evopool_no_codream" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e15/

echo "=== Done: evopool_no_codream | $(date) ==="
