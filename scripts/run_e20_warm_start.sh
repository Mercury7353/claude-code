#!/bin/bash
#SBATCH -J evopool_e20_warm
#SBATCH -A hw-grp
#SBATCH -p hw-grp
#SBATCH --gres=gpu:0
#SBATCH -t 8:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -o /nfs/hpc/share/zhanyaol/claude-code/logs/e20_warm_start_%j.out
#SBATCH -e /nfs/hpc/share/zhanyaol/claude-code/logs/e20_warm_start_%j.err

# E20: EvoPool warm-start — re-run AFlow-Stream with the fully-evolved E17 pool.
# Tests whether CoDream's accumulated memories are genuinely useful across a second pass.
# Design: identical setup to E17, but pool agents carry forward all L1/L2/L3 memories
# from the first 600 tasks.  Fresh cold-start pool (E17) is the comparison.
#
# Expected result: higher Q1 scores across all domains (pre-loaded insights),
# less within-domain growth needed, potentially higher overall accuracy.
echo "=== E20: EvoPool Warm-Start (E17 pool) ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "Git: $(git -C /nfs/hpc/share/zhanyaol/claude-code log --oneline -1)"

source /nfs/hpc/share/zhanyaol/miniconda3/etc/profile.d/conda.sh
conda activate tool
cd /nfs/hpc/share/zhanyaol/claude-code

POOL_FILE="results/e21/evopool_pool_state.json"
if [ ! -f "$POOL_FILE" ]; then
    echo "ERROR: E21 pool state not found at $POOL_FILE"
    echo "Run E21 (run_e21_save_pool.sh) first."
    exit 1
fi
echo "Loading evolved pool from: $POOL_FILE"

URL6=""
URL7=""
check_server() {
    local url="$1"
    curl -s --max-time 5 "${url}/v1/models" | grep -q qwen 2>/dev/null
}
for jf in "vllm_server_8007.json" "vllm_server_6.json"; do
    [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null)
    check_server "$candidate" && URL6="$candidate" && echo "Server A: $URL6 (from $jf)" && break
done
for jf in "vllm_server_8008.json" "vllm_server_7.json"; do
    [ -f "$jf" ] && candidate=$(python3 -c "import json; d=json.load(open('$jf')); print(d['url'])" 2>/dev/null)
    check_server "$candidate" && URL7="$candidate" && echo "Server B: $URL7 (from $jf)" && break
done

if [ -z "$URL6" ] || [ -z "$URL7" ]; then
    echo "ERROR: No responsive vLLM servers found"
    exit 1
fi
echo "Using servers: $URL6, $URL7"
export EVOPOOL_LOCAL_LLM_URLS="$URL6,$URL7"
export HF_DATASETS_OFFLINE=1
mkdir -p results/e20

python -u run_experiment.py \
    --condition "evopool_full" \
    --benchmark aflow_stream \
    --n_per_domain 100 \
    --pool_size 20 \
    --team_size 3 \
    --backbone_llm "qwen3-8b" \
    --seed 42 \
    --load_pool "$POOL_FILE" \
    --output_dir results/e20/

echo "=== Done: EvoPool warm-start | $(date) ==="
