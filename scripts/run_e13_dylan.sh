#!/bin/bash
#SBATCH -J evopool_e13_dylan
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e13_dylan_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e13_dylan_%j.err

# E13: Clean DyLAN rerun with thinking mode disabled (fix utils.py) + servers 6/7
echo "=== EvoPool E13 DyLAN (thinking disabled, servers 6/7) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

# Wait for servers 6/7 to be healthy (with timeout)
MAX_WAIT=900  # 15 minutes
WAIT=0
echo "Waiting for vllm_server_6.json and vllm_server_7.json..."
while [ $WAIT -lt $MAX_WAIT ]; do
    URL6=""
    URL7=""
    [ -f "vllm_server_6.json" ] && URL6=$(python3 -c "import json; d=json.load(open('vllm_server_6.json')); print(d['url'])" 2>/dev/null)
    [ -f "vllm_server_7.json" ] && URL7=$(python3 -c "import json; d=json.load(open('vllm_server_7.json')); print(d['url'])" 2>/dev/null)

    H6=0; H7=0
    if [ -n "$URL6" ]; then
        curl -sf "$URL6/health" >/dev/null 2>&1 && H6=1
    fi
    if [ -n "$URL7" ]; then
        curl -sf "$URL7/health" >/dev/null 2>&1 && H7=1
    fi

    [ $H6 -eq 1 ] && [ $H7 -eq 1 ] && break
    echo "  Waiting... ($WAIT s) health: server6=$H6 server7=$H7"
    sleep 30
    WAIT=$((WAIT + 30))
done

if [ -z "$URL6" ] || [ -z "$URL7" ]; then
    echo "ERROR: Servers 6/7 not ready after ${MAX_WAIT}s"
    exit 1
fi
echo "Servers ready: $URL6, $URL7"
export EVOPOOL_LOCAL_LLM_URLS="$URL6,$URL7"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e13

python -u run_experiment.py \
    --condition "dylan" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --output_dir results/e13/

echo "=== Done: DyLAN fix | $(date) ==="
